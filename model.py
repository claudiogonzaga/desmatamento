"""
ABM de Desmatamento Ilegal e Dissuasão via ACP Ambiental
Modelo baseado em Mesa — Teoria Econômica do Crime (Becker, 1968)
e modelo de dissuasão de Schmitt (2015), adaptado para a via judicial.
"""

import warnings
warnings.filterwarnings("ignore", message=".*place_agent.*")

import numpy as np
import mesa
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from math import exp
from collections import deque


# ── Estados da célula ────────────────────────────────────────────────
FOREST = 0
PASTURE = 1
SOY = 2
DEGRADED_PASTURE = 3
REGENERATING = 4
APP = 5
LEGAL_RESERVE = 6
WATER = 7
ROAD = 8
EMBARGOED = 9

STATE_NAMES = {
    FOREST: "Floresta",
    PASTURE: "Pastagem",
    SOY: "Soja",
    DEGRADED_PASTURE: "Past. degradada",
    REGENERATING: "Em regeneração",
    APP: "APP",
    LEGAL_RESERVE: "Reserva Legal",
    WATER: "Água",
    ROAD: "Estrada",
    EMBARGOED: "Embargada",
}


# ── Agente célula (parcela de 1 ha) ─────────────────────────────────
class LandCell(Agent):
    """Cada célula do grid = 1 hectare."""

    def __init__(self, unique_id, model, pos, state=FOREST):
        super().__init__(unique_id, model)
        self._init_pos = pos
        self.state = state
        self.owner_id = None
        self.years_degraded = 0
        self.years_regenerating = 0
        self.embargo_year = None
        self.converted_year = None

    def step(self):
        if self.state == PASTURE:
            owner = self._get_owner()
            degrade_prob = 0.02 if owner and owner.profile == "sojicultor" else 0.04
            if self.random.random() < degrade_prob:
                self.state = DEGRADED_PASTURE
                self.years_degraded = 0

        if self.state == DEGRADED_PASTURE:
            self.years_degraded += 1
            if self.years_degraded > 15:
                if self.random.random() < 0.15:
                    self.state = REGENERATING
                    self.years_regenerating = 0

        if self.state == REGENERATING:
            self.years_regenerating += 1
            if self.years_regenerating >= 20:
                self.state = FOREST

    def _get_owner(self):
        for agent in self.model.schedule.agents:
            if isinstance(agent, Landholder) and agent.unique_id == self.owner_id:
                return agent
        return None


# ── Agente proprietário rural ────────────────────────────────────────
class Landholder(Agent):
    """Proprietário rural — decide se converte floresta a cada ano."""

    PROFILES = {
        "pequeno": {
            "size_range": (20, 100),
            "alpha_range": (0.6, 0.9),
            "delta_range": (0.12, 0.18),
            "uso_principal": "pecuaria",
            "convert_rate": (1, 5),
        },
        "grande_pecuarista": {
            "size_range": (200, 2000),
            "alpha_range": (0.3, 0.6),
            "delta_range": (0.08, 0.12),
            "uso_principal": "pecuaria",
            "convert_rate": (5, 40),
        },
        "sojicultor": {
            "size_range": (500, 5000),
            "alpha_range": (0.2, 0.5),
            "delta_range": (0.06, 0.10),
            "uso_principal": "soja",
            "convert_rate": (10, 80),
        },
    }

    def __init__(self, unique_id, model, profile, cells, prop_rect):
        super().__init__(unique_id, model)
        self.profile = profile
        self.cells = cells
        self.cells_set = set(cells)
        self.prop_rect = prop_rect  # (x0, y0, x1, y1)
        cfg = self.PROFILES[profile]
        self.alpha = self.random.uniform(*cfg["alpha_range"])
        self.delta = self.random.uniform(*cfg["delta_range"])
        self.uso_principal = cfg["uso_principal"]
        self.convert_min, self.convert_max = cfg["convert_rate"]
        self.wealth = 0.0
        self.n_infractions = 0
        self.n_acps = 0
        self.embargoed = False
        self.embargo_year = None
        self.total_converted = 0
        self.acp_pending_values = []

    @property
    def total_cells(self):
        return len(self.cells)

    def _count_state(self, state):
        count = 0
        for pos in self.cells:
            cell = self._cell_at(pos)
            if cell and cell.state == state:
                count += 1
        return count

    def _cell_at(self, pos):
        contents = self.model.grid.get_cell_list_contents([pos])
        for c in contents:
            if isinstance(c, LandCell):
                return c
        return None

    @property
    def forest_pct(self):
        if self.total_cells == 0:
            return 0
        return self._count_state(FOREST) / self.total_cells

    @property
    def forest_cells_count(self):
        return self._count_state(FOREST)

    @property
    def legal_reserve_cells(self):
        return self._count_state(LEGAL_RESERVE)

    @property
    def legal_reserve_target(self):
        return 0.80

    @property
    def convertible_forest(self):
        total = self.total_cells
        forest = self.forest_cells_count
        lr = self.legal_reserve_cells
        protected = int(total * self.legal_reserve_target)
        available = forest - max(0, protected - lr)
        return max(0, available)

    def _neighbor_conversion_pressure(self):
        """Fração de vizinhos já convertidos (efeito fronteira) — amostragem."""
        # Amostra para performance
        sample_cells = self.cells if len(self.cells) < 50 else \
            self.random.sample(self.cells, 50)
        converted = 0
        total_neighbors = 0
        for pos in sample_cells:
            neighbors = self.model.grid.get_neighborhood(
                pos, moore=True, include_center=False
            )
            for npos in neighbors:
                cell = self.model._cells.get(npos)
                if cell:
                    total_neighbors += 1
                    if cell.state in (PASTURE, SOY):
                        converted += 1
        if total_neighbors == 0:
            return 0
        return converted / total_neighbors

    def _calc_VE(self):
        p = self.model.params
        Gf = p["Gf"]
        Cp = p["Cp"]
        Gt = p["Gt"]
        if self.uso_principal == "soja":
            G_uso = p["Ga"]
        else:
            G_uso = p["Gp"]
        return Gf + (G_uso * Cp) + Gt

    def _calc_VD_admin(self):
        p = self.model.params
        return (p["Pd"] * p["Pa"] * p["Pj"] * p["Pc"] * p["Pp"]
                * (p["S"] + p["Ve"] + p["Va"])
                * exp(-p["r"] * p["t_admin"]))

    def _calc_VD_acp(self):
        p = self.model.params
        return (p["Pd"] * p["P_inq"] * p["P_acp"] * p["P_cond"] * p["P_exec"]
                * p["V_ACP"]
                * exp(-p["r"] * p["t_acp"]))

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + exp(-x))

    def step(self):
        p = self.model.params

        # Receita anual
        n_pasture = self._count_state(PASTURE)
        n_soy = self._count_state(SOY)
        self.wealth += n_pasture * p["Gp"] + n_soy * p["Ga"]

        # ACPs pendentes
        new_pending = []
        for valor, ano in self.acp_pending_values:
            elapsed = self.model.year - ano
            if elapsed >= p["t_acp"]:
                if self.random.random() < p["P_exec"]:
                    self.wealth -= valor
            else:
                new_pending.append((valor, ano))
        self.acp_pending_values = new_pending

        # Desembargo
        if self.embargoed and self.embargo_year is not None:
            if self.model.year - self.embargo_year >= 3:
                self.embargoed = False
                self.embargo_year = None
                for pos in self.cells:
                    cell = self._cell_at(pos)
                    if cell and cell.state == EMBARGOED:
                        cell.state = DEGRADED_PASTURE

        # Decisão de conversão
        if self.embargoed:
            if self.random.random() > (1 - p["Pd"]) * 0.3:
                return

        available = self.convertible_forest
        if available <= 0:
            return

        VE = self._calc_VE()
        VD_admin = self._calc_VD_admin()
        VD_acp = self._calc_VD_acp()
        VD_total = VD_admin + VD_acp
        c = p["c_desmat"]
        C = VE - (VD_total + c)

        neighbor_pressure = self._neighbor_conversion_pressure()
        C_adjusted = C + neighbor_pressure * 500

        prob = self._sigmoid(C_adjusted / (1000 * self.alpha))
        if self.n_infractions > 0:
            prob = min(1.0, prob * 1.1)

        if self.random.random() < prob:
            max_conv = min(self.convert_max, available)
            if max_conv <= 0:
                return
            min_conv = min(self.convert_min, max_conv)
            n_convert = self.random.randint(min_conv, max_conv)
            if n_convert <= 0:
                return

            actually_converted = self._convert_cells_contiguous(n_convert)
            self.model.annual_converted += actually_converted
            self.total_converted += actually_converted

            if actually_converted > 0:
                self._apply_enforcement(actually_converted)

    def _convert_cells_contiguous(self, n):
        """
        Converte n hectares como BLOCO CONTÍGUO, expandindo a partir
        da borda existente de pastagem/soja/estrada (como desmatamento real).
        Usa BFS (flood-fill) a partir das sementes de borda.
        """
        target_state = SOY if self.uso_principal == "soja" else PASTURE

        # Encontrar sementes: células de floresta adjacentes a área já convertida/estrada
        seeds = []
        for pos in self.cells:
            cell = self.model._cells.get(pos)
            if cell and cell.state == FOREST:
                # Verificar se é borda (adjacente a convertido)
                neighbors = self.model.grid.get_neighborhood(
                    pos, moore=False, include_center=False  # Von Neumann = 4 vizinhos
                )
                is_edge = False
                for npos in neighbors:
                    ncell = self.model._cells.get(npos)
                    if ncell and ncell.state in (PASTURE, SOY, ROAD, DEGRADED_PASTURE, EMBARGOED):
                        is_edge = True
                        break
                if is_edge:
                    seeds.append(pos)

        # Se não há sementes de borda, pegar floresta mais próxima da estrada
        if not seeds:
            x0, y0, x1, y1 = self.prop_rect
            road_y = self.model.params["grid_size"] // 2
            road_x = self.model.params["grid_size"] // 2
            forest_cells = []
            for pos in self.cells:
                cell = self.model._cells.get(pos)
                if cell and cell.state == FOREST:
                    dist = min(abs(pos[0] - road_x), abs(pos[1] - road_y))
                    forest_cells.append((pos, dist))
            if not forest_cells:
                return 0
            forest_cells.sort(key=lambda x: x[1])
            seeds = [forest_cells[0][0]]

        # BFS a partir das sementes — expande em bloco contíguo
        queue = deque(seeds)
        visited = set(seeds)
        to_convert = []

        while queue and len(to_convert) < n:
            pos = queue.popleft()
            cell = self.model._cells.get(pos)
            if cell and cell.state == FOREST and pos in self.cells_set:
                to_convert.append(pos)
                # Adicionar vizinhos (4-vizinhança para blocos mais compactos)
                neighbors = self.model.grid.get_neighborhood(
                    pos, moore=False, include_center=False
                )
                for npos in neighbors:
                    if npos not in visited and npos in self.cells_set:
                        ncell = self.model._cells.get(npos)
                        if ncell and ncell.state == FOREST:
                            visited.add(npos)
                            queue.append(npos)

        # Converter
        for pos in to_convert:
            cell = self.model._cells.get(pos)
            if cell:
                cell.state = target_state
                cell.converted_year = self.model.year

        return len(to_convert)

    def _apply_enforcement(self, n_convert):
        """Fiscalização após conversão."""
        p = self.model.params

        if self.random.random() < p["Pd"]:
            self.model.annual_detected += n_convert
            if self.random.random() < p["Pa"]:
                self.n_infractions += 1
                self.model.annual_infractions += 1
                if self.random.random() < 0.7:
                    self.embargoed = True
                    self.embargo_year = self.model.year
                    self.model.annual_embargoes += 1
                    self._embargo_converted(n_convert)

                p_inq_eff = min(p["P_inq"] * (2.0 if n_convert > 50 else 1.0), 1.0)
                if self.random.random() < p_inq_eff:
                    if self.random.random() < p["P_acp"]:
                        if self.random.random() < p["P_cond"]:
                            valor_acp = n_convert * p["V_ACP"]
                            self.n_acps += 1
                            self.acp_pending_values.append(
                                (valor_acp, self.model.year)
                            )
                            self.model.annual_acps += 1

    def _embargo_converted(self, n):
        count = 0
        for pos in self.cells:
            cell = self._cell_at(pos)
            if cell and cell.state in (PASTURE, SOY) and cell.converted_year == self.model.year:
                cell.state = EMBARGOED
                cell.embargo_year = self.model.year
                count += 1
                if count >= n:
                    break


# ── Modelo principal ─────────────────────────────────────────────────
class DeforestationModel(Model):
    """Modelo ABM de desmatamento ilegal em município da Amazônia Legal."""

    DEFAULT_PARAMS = {
        "Pd": 0.55, "Pa": 0.30, "Pj": 0.26, "Pc": 0.90, "Pp": 0.10,
        "S": 5000.0, "Ve": 300.0, "Va": 15185.22, "t_admin": 2.9,
        "P_inq": 0.08, "P_acp": 0.40, "P_cond": 0.65, "P_exec": 0.30,
        "V_ACP": 15000.0, "t_acp": 7.0,
        "Gf": 2000.0, "Gp": 300.0, "Ga": 500.0, "Gt": 5000.0,
        "Cp": 5, "c_desmat": 200.0, "r": 0.1375,
        "grid_size": 100, "n_properties": 40, "n_years": 15, "seed": 42,
    }

    def __init__(self, params=None):
        super().__init__()
        self.params = dict(self.DEFAULT_PARAMS)
        if params:
            self.params.update(params)

        self.random.seed(self.params["seed"])
        np.random.seed(self.params["seed"])

        N = self.params["grid_size"]
        self.grid = MultiGrid(N, N, torus=False)
        self.schedule = RandomActivation(self)

        self.year = 0
        self.annual_converted = 0
        self.annual_detected = 0
        self.annual_infractions = 0
        self.annual_embargoes = 0
        self.annual_acps = 0
        self.history = []

        # Criar células
        cell_id = 0
        self._cells = {}
        for x in range(N):
            for y in range(N):
                cell = LandCell(cell_id, self, (x, y), state=FOREST)
                self.grid.place_agent(cell, (x, y))
                self.schedule.add(cell)
                self._cells[(x, y)] = cell
                cell_id += 1

        # Estradas
        self._create_roads(N)

        # Propriedades
        self._agent_id_counter = cell_id
        self.landholders = []
        self._create_properties(N)

        # Reserva legal no fundo da propriedade (longe das estradas)
        self._assign_legal_reserves()

        # DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                "year": lambda m: m.year,
                "pct_forest": lambda m: m._pct_state(FOREST),
                "pct_pasture": lambda m: m._pct_state(PASTURE),
                "pct_soy": lambda m: m._pct_state(SOY),
                "pct_degraded": lambda m: m._pct_state(DEGRADED_PASTURE),
                "pct_regenerating": lambda m: m._pct_state(REGENERATING),
                "pct_embargoed": lambda m: m._pct_state(EMBARGOED),
                "pct_legal_reserve": lambda m: m._pct_state(LEGAL_RESERVE),
                "pct_app": lambda m: m._pct_state(APP),
                "annual_converted_ha": lambda m: m.annual_converted,
                "annual_infractions": lambda m: m.annual_infractions,
                "annual_embargoes": lambda m: m.annual_embargoes,
                "annual_acps": lambda m: m.annual_acps,
                "VD_admin_mean": lambda m: m._mean_VD_admin(),
                "VD_acp_mean": lambda m: m._mean_VD_acp(),
                "VD_total_mean": lambda m: m._mean_VD_total(),
                "VE_mean": lambda m: m._mean_VE(),
                "C_mean": lambda m: m._mean_C(),
                "n_embargoed_props": lambda m: sum(
                    1 for lh in m.landholders if lh.embargoed
                ),
                "total_wealth": lambda m: sum(
                    lh.wealth for lh in m.landholders
                ),
            }
        )

    def _pct_state(self, state):
        N = self.params["grid_size"]
        total = N * N
        count = sum(1 for cell in self._cells.values() if cell.state == state)
        return count / total * 100

    def _mean_VD_admin(self):
        if not self.landholders:
            return 0
        return np.mean([lh._calc_VD_admin() for lh in self.landholders])

    def _mean_VD_acp(self):
        if not self.landholders:
            return 0
        return np.mean([lh._calc_VD_acp() for lh in self.landholders])

    def _mean_VD_total(self):
        return self._mean_VD_admin() + self._mean_VD_acp()

    def _mean_VE(self):
        if not self.landholders:
            return 0
        return np.mean([lh._calc_VE() for lh in self.landholders])

    def _mean_C(self):
        return self._mean_VE() - (self._mean_VD_total() + self.params["c_desmat"])

    def _create_roads(self, N):
        """Estradas em cruz."""
        mid = N // 2
        for i in range(N):
            for pos in [(mid, i), (i, mid)]:
                cell = self._cells.get(pos)
                if cell:
                    cell.state = ROAD

    def _create_properties(self, N):
        np_rng = np.random.RandomState(self.params["seed"])
        n_props = self.params["n_properties"]

        n_pequeno = max(1, int(n_props * 0.45))
        n_grande = max(1, int(n_props * 0.35))
        n_soja = max(1, n_props - n_pequeno - n_grande)

        profiles = (
            ["pequeno"] * n_pequeno
            + ["grande_pecuarista"] * n_grande
            + ["sojicultor"] * n_soja
        )
        np_rng.shuffle(profiles)

        rects = self._partition_grid(N, n_props, np_rng)

        for i, (profile, rect) in enumerate(zip(profiles, rects)):
            x0, y0, x1, y1 = rect
            cells = []
            for x in range(x0, x1):
                for y in range(y0, y1):
                    if (x, y) in self._cells:
                        cells.append((x, y))

            if not cells:
                continue

            agent_id = self._agent_id_counter
            self._agent_id_counter += 1
            lh = Landholder(agent_id, self, profile, cells, rect)
            self.schedule.add(lh)
            self.landholders.append(lh)

            for pos in cells:
                cell = self._cells.get(pos)
                if cell:
                    cell.owner_id = agent_id

    def _partition_grid(self, N, n_parts, rng):
        rects = [(0, 0, N, N)]
        while len(rects) < n_parts:
            rects.sort(key=lambda r: -(r[2]-r[0]) * (r[3]-r[1]))
            rect = rects.pop(0)
            x0, y0, x1, y1 = rect
            w = x1 - x0
            h = y1 - y0
            if w <= 2 and h <= 2:
                rects.append(rect)
                break
            if w >= h and w > 2:
                split = x0 + max(1, int(w * rng.uniform(0.3, 0.7)))
                rects.append((x0, y0, split, y1))
                rects.append((split, y0, x1, y1))
            elif h > 2:
                split = y0 + max(1, int(h * rng.uniform(0.3, 0.7)))
                rects.append((x0, y0, x1, split))
                rects.append((x0, split, x1, y1))
            else:
                rects.append(rect)
                break
        return rects[:n_parts]

    def _assign_legal_reserves(self):
        """
        Reserva legal no FUNDO da propriedade (longe das estradas).
        As estradas cruzam no centro do grid — então a RL fica nos cantos.
        """
        road_x = self.params["grid_size"] // 2
        road_y = self.params["grid_size"] // 2

        for lh in self.landholders:
            forest_in_prop = []
            for pos in lh.cells:
                cell = self._cells.get(pos)
                if cell and cell.state == FOREST:
                    # Distância Manhattan da estrada mais próxima
                    dist = min(abs(pos[0] - road_x), abs(pos[1] - road_y))
                    forest_in_prop.append((pos, dist))

            if not forest_in_prop:
                continue

            # Ordenar por distância DECRESCENTE — mais longe = fundo = RL
            forest_in_prop.sort(key=lambda x: -x[1])

            # 55% da propriedade como RL (irregular, mas cria dinâmica)
            n_lr = int(len(forest_in_prop) * 0.55)
            for pos, _ in forest_in_prop[:n_lr]:
                cell = self._cells.get(pos)
                if cell:
                    cell.state = LEGAL_RESERVE

    def get_grid_array(self):
        N = self.params["grid_size"]
        arr = np.zeros((N, N), dtype=int)
        for (x, y), cell in self._cells.items():
            arr[y, x] = cell.state
        return arr

    def get_property_boundaries(self):
        boundaries = []
        for lh in self.landholders:
            if not lh.cells:
                continue
            xs = [p[0] for p in lh.cells]
            ys = [p[1] for p in lh.cells]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            boundaries.append({
                "x0": x0, "y0": y0,
                "w": x1 - x0 + 1, "h": y1 - y0 + 1,
                "embargoed": lh.embargoed,
                "profile": lh.profile,
                "id": lh.unique_id,
            })
        return boundaries

    def _make_snapshot(self):
        return {
            "year": self.year,
            "grid": self.get_grid_array().copy(),
            "pct_forest": self._pct_state(FOREST),
            "pct_pasture": self._pct_state(PASTURE),
            "pct_soy": self._pct_state(SOY),
            "pct_degraded": self._pct_state(DEGRADED_PASTURE),
            "pct_embargoed": self._pct_state(EMBARGOED),
            "pct_regenerating": self._pct_state(REGENERATING),
            "annual_converted": self.annual_converted,
            "annual_infractions": self.annual_infractions,
            "annual_embargoes": self.annual_embargoes,
            "annual_acps": self.annual_acps,
            "VD_admin": self._mean_VD_admin(),
            "VD_acp": self._mean_VD_acp(),
            "VD_total": self._mean_VD_total(),
            "VE": self._mean_VE(),
            "C": self._mean_C(),
            "boundaries": self.get_property_boundaries(),
        }

    def step(self):
        self.year += 1
        self.annual_converted = 0
        self.annual_detected = 0
        self.annual_infractions = 0
        self.annual_embargoes = 0
        self.annual_acps = 0

        self.schedule.step()
        self.datacollector.collect(self)
        self.history.append(self._make_snapshot())

    def run(self, n_years=None):
        if n_years is None:
            n_years = self.params["n_years"]
        self.datacollector.collect(self)
        self.history.append(self._make_snapshot())
        for _ in range(n_years):
            self.step()
        return self.datacollector.get_model_vars_dataframe()
