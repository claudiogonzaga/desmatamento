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
        self._init_pos = pos  # pos will be set by place_agent
        self.state = state
        self.owner_id = None          # ID do proprietário
        self.years_degraded = 0       # anos em degradação
        self.years_regenerating = 0   # anos em regeneração
        self.embargo_year = None      # ano em que foi embargada
        self.converted_year = None    # ano da conversão

    def step(self):
        # Degradação natural de pastagem sem manejo
        if self.state == PASTURE:
            owner = self._get_owner()
            degrade_prob = 0.02 if owner and owner.profile == "sojicultor" else 0.04
            if self.random.random() < degrade_prob:
                self.state = DEGRADED_PASTURE
                self.years_degraded = 0

        # Contagem de degradação → regeneração se abandonada
        if self.state == DEGRADED_PASTURE:
            self.years_degraded += 1
            if self.years_degraded > 15:
                if self.random.random() < 0.15:
                    self.state = REGENERATING
                    self.years_regenerating = 0

        # Regeneração lenta
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
            "convert_rate": (1, 5),   # hectares por ano se decidir
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

    def __init__(self, unique_id, model, profile, cells):
        super().__init__(unique_id, model)
        self.profile = profile
        self.cells = cells  # lista de (x, y)
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
        self.acp_pending_values = []   # (valor, ano_ajuizamento)

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
        """80% na Amazônia Legal."""
        return 0.80

    @property
    def convertible_forest(self):
        """Floresta que pode ser convertida (acima da reserva legal)."""
        total = self.total_cells
        forest = self.forest_cells_count
        lr = self.legal_reserve_cells
        protected = int(total * self.legal_reserve_target)
        available = forest - max(0, protected - lr)
        return max(0, available)

    def _neighbor_conversion_pressure(self):
        """Fração de vizinhos já convertidos (efeito fronteira)."""
        converted = 0
        total_neighbors = 0
        for pos in self.cells:
            neighbors = self.model.grid.get_neighborhood(pos, moore=True, include_center=False)
            for npos in neighbors:
                contents = self.model.grid.get_cell_list_contents([npos])
                for c in contents:
                    if isinstance(c, LandCell):
                        total_neighbors += 1
                        if c.state in (PASTURE, SOY):
                            converted += 1
        if total_neighbors == 0:
            return 0
        return converted / total_neighbors

    def _calc_VE(self):
        """Vantagem econômica do desmatamento por hectare."""
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
        """Dissuasão administrativa (Schmitt)."""
        p = self.model.params
        Pd = p["Pd"]
        Pa = p["Pa"]
        Pj = p["Pj"]
        Pc = p["Pc"]
        Pp = p["Pp"]
        S = p["S"]
        Ve = p["Ve"]
        Va = p["Va"]
        r = p["r"]
        t = p["t_admin"]
        return Pd * Pa * Pj * Pc * Pp * (S + Ve + Va) * exp(-r * t)

    def _calc_VD_acp(self):
        """Dissuasão via ACP ambiental."""
        p = self.model.params
        Pd = p["Pd"]
        P_inq = p["P_inq"]
        P_acp = p["P_acp"]
        P_cond = p["P_cond"]
        P_exec = p["P_exec"]
        V_ACP = p["V_ACP"]
        r = p["r"]
        t_acp = p["t_acp"]
        return Pd * P_inq * P_acp * P_cond * P_exec * V_ACP * exp(-r * t_acp)

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + exp(-x))

    def step(self):
        p = self.model.params

        # Receita anual das áreas já convertidas
        n_pasture = self._count_state(PASTURE)
        n_soy = self._count_state(SOY)
        self.wealth += n_pasture * p["Gp"] + n_soy * p["Ga"]

        # Custos de ACPs pendentes (pagamento eventual)
        new_pending = []
        for valor, ano in self.acp_pending_values:
            elapsed = self.model.year - ano
            if elapsed >= p["t_acp"]:
                # Execução
                if self.random.random() < p["P_exec"]:
                    self.wealth -= valor
            else:
                new_pending.append((valor, ano))
        self.acp_pending_values = new_pending

        # Desembargo após 3 anos se não reincidiu
        if self.embargoed and self.embargo_year is not None:
            if self.model.year - self.embargo_year >= 3:
                self.embargoed = False
                self.embargo_year = None
                for pos in self.cells:
                    cell = self._cell_at(pos)
                    if cell and cell.state == EMBARGOED:
                        cell.state = DEGRADED_PASTURE

        # ── Decisão de conversão ──
        if self.embargoed:
            # Pode descumprir embargo com baixa probabilidade
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

        # Efeito vizinhança
        neighbor_pressure = self._neighbor_conversion_pressure()
        C_adjusted = C + neighbor_pressure * 500  # bonus vizinhança

        # Aversão ao risco modera a decisão
        prob = self._sigmoid(C_adjusted / (1000 * self.alpha))

        # Reincidentes ficam mais ousados
        if self.n_infractions > 0:
            prob = min(1.0, prob * 1.1)

        if self.random.random() < prob:
            # Decide converter
            max_conv = min(self.convert_max, available)
            if max_conv <= 0:
                return
            min_conv = min(self.convert_min, max_conv)
            n_convert = self.random.randint(min_conv, max_conv)
            if n_convert <= 0:
                return

            self._convert_cells(n_convert)
            self.model.annual_converted += n_convert
            self.total_converted += n_convert

            # ── Fiscalização ──
            # Detecção
            if self.random.random() < p["Pd"]:
                self.model.annual_detected += n_convert
                # Autuação
                if self.random.random() < p["Pa"]:
                    self.n_infractions += 1
                    self.model.annual_infractions += 1
                    multa = n_convert * p["S"]
                    # Embargo
                    if self.random.random() < 0.7:
                        self.embargoed = True
                        self.embargo_year = self.model.year
                        self.model.annual_embargoes += 1
                        self._embargo_converted(n_convert)

                    # MP abre inquérito? (mais provável para grandes)
                    p_inq_eff = p["P_inq"]
                    if n_convert > 50:
                        p_inq_eff *= 2.0
                    p_inq_eff = min(p_inq_eff, 1.0)

                    if self.random.random() < p_inq_eff:
                        # ACP ajuizada?
                        if self.random.random() < p["P_acp"]:
                            # Condenação?
                            if self.random.random() < p["P_cond"]:
                                valor_acp = n_convert * p["V_ACP"]
                                self.n_acps += 1
                                self.acp_pending_values.append(
                                    (valor_acp, self.model.year)
                                )
                                self.model.annual_acps += 1

    def _convert_cells(self, n):
        """Converte n hectares de floresta, priorizando bordas."""
        forest_cells = []
        for pos in self.cells:
            cell = self._cell_at(pos)
            if cell and cell.state == FOREST:
                # Score: prioriza bordas com áreas já convertidas
                neighbors = self.model.grid.get_neighborhood(
                    pos, moore=True, include_center=False
                )
                edge_score = 0
                for npos in neighbors:
                    contents = self.model.grid.get_cell_list_contents([npos])
                    for nc in contents:
                        if isinstance(nc, LandCell) and nc.state in (PASTURE, SOY, ROAD):
                            edge_score += 1
                forest_cells.append((pos, edge_score))

        # Ordena por score de borda (prioriza fronteira)
        forest_cells.sort(key=lambda x: -x[1])

        target_state = SOY if self.uso_principal == "soja" else PASTURE
        converted = 0
        for pos, _ in forest_cells:
            if converted >= n:
                break
            cell = self._cell_at(pos)
            if cell:
                cell.state = target_state
                cell.converted_year = self.model.year
                converted += 1

    def _embargo_converted(self, n):
        """Marca células recém-convertidas como embargadas."""
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
        # Fiscalização administrativa (Schmitt 2015, atualizado)
        "Pd": 0.55,      # prob. detecção
        "Pa": 0.30,       # prob. autuação
        "Pj": 0.26,       # prob. julgamento 1ª inst.
        "Pc": 0.90,       # prob. confirmação
        "Pp": 0.10,       # prob. pagamento
        "S": 5000.0,      # multa R$/ha
        "Ve": 300.0,      # valor embargo (lucro cessante)
        "Va": 15185.22,   # valor bens apreendidos
        "t_admin": 2.9,   # tempo médio julgamento admin (anos)
        # ACP (novo)
        "P_inq": 0.08,    # prob. inquérito MP
        "P_acp": 0.40,    # prob. ACP ajuizada
        "P_cond": 0.65,   # prob. condenação
        "P_exec": 0.30,   # prob. execução sentença
        "V_ACP": 15000.0, # valor ACP R$/ha
        "t_acp": 7.0,     # tempo até execução (anos)
        # Economia
        "Gf": 2000.0,     # ganho floresta ilegal
        "Gp": 300.0,      # ganho pecuária R$/ha/ano
        "Ga": 500.0,      # ganho soja R$/ha/ano
        "Gt": 5000.0,     # ganho valorização terra
        "Cp": 5,          # coef. prescrição (anos)
        "c_desmat": 200.0, # custo desmatamento R$/ha
        "r": 0.1375,      # taxa Selic
        # Grid e agentes
        "grid_size": 100,
        "n_properties": 40,
        "n_years": 15,
        "seed": 42,
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

        # Criar rede de estradas (simples: cruz central + bordas)
        self._create_roads(N)

        # Criar corpos d'água
        self._create_water(N)

        # Criar APP ao redor da água
        self._create_app(N)

        # Criar propriedades
        self._agent_id_counter = cell_id
        self.landholders = []
        self._create_properties(N)

        # Marcar reservas legais nas propriedades
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
        count = sum(
            1 for cell in self._cells.values()
            if cell.state == state
        )
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
        """Estradas em cruz + bordas."""
        mid = N // 2
        for i in range(N):
            for pos in [(mid, i), (i, mid)]:
                cell = self._cells.get(pos)
                if cell:
                    cell.state = ROAD
            # Bordas
            for pos in [(0, i), (N-1, i), (i, 0), (i, N-1)]:
                cell = self._cells.get(pos)
                if cell and self.random.random() < 0.3:
                    cell.state = ROAD

    def _create_water(self, N):
        """Rio sinuoso."""
        y = N // 3
        for x in range(N):
            dy = int(3 * np.sin(x * 2 * np.pi / N))
            for w in range(-1, 2):
                yw = y + dy + w
                if 0 <= yw < N:
                    cell = self._cells.get((x, yw))
                    if cell and cell.state != ROAD:
                        cell.state = WATER

    def _create_app(self, N):
        """APP: 30m (≈1 célula) ao redor de água."""
        water_cells = [
            pos for pos, cell in self._cells.items()
            if cell.state == WATER
        ]
        for pos in water_cells:
            neighbors = self.grid.get_neighborhood(pos, moore=True, include_center=False)
            for npos in neighbors:
                cell = self._cells.get(npos)
                if cell and cell.state == FOREST:
                    cell.state = APP

    def _create_properties(self, N):
        """Particiona o grid em propriedades retangulares."""
        np_rng = np.random.RandomState(self.params["seed"])
        n_props = self.params["n_properties"]

        # Distribuição de perfis
        n_pequeno = max(1, int(n_props * 0.45))
        n_grande = max(1, int(n_props * 0.35))
        n_soja = max(1, n_props - n_pequeno - n_grande)

        profiles = (
            ["pequeno"] * n_pequeno
            + ["grande_pecuarista"] * n_grande
            + ["sojicultor"] * n_soja
        )
        np_rng.shuffle(profiles)

        # Particionar grid em retângulos via subdivisão recursiva
        available = set()
        for x in range(N):
            for y in range(N):
                cell = self._cells.get((x, y))
                if cell and cell.state == FOREST:
                    available.add((x, y))

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
            lh = Landholder(agent_id, self, profile, cells)
            self.schedule.add(lh)
            self.landholders.append(lh)

            for pos in cells:
                cell = self._cells.get(pos)
                if cell:
                    cell.owner_id = agent_id

    def _partition_grid(self, N, n_parts, rng):
        """Subdivide grid em retângulos aproximadamente iguais."""
        rects = [(0, 0, N, N)]

        while len(rects) < n_parts:
            # Pega o maior retângulo e divide
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
        """Marca 80% da floresta de cada propriedade como Reserva Legal."""
        for lh in self.landholders:
            forest_in_prop = []
            for pos in lh.cells:
                cell = self._cells.get(pos)
                if cell and cell.state == FOREST:
                    forest_in_prop.append(pos)

            if not forest_in_prop:
                continue

            # 80% da propriedade deve ser RL
            # Mas queremos que parte seja convertível (senão não há dinâmica)
            # Na prática, marcamos ~60% como RL (o que já está irregular por definição
            # visto que a lei exige 80%)
            n_lr = int(len(forest_in_prop) * 0.55)
            # Marca as mais internas como RL
            self.random.shuffle(forest_in_prop)
            for pos in forest_in_prop[:n_lr]:
                cell = self._cells.get(pos)
                if cell:
                    cell.state = LEGAL_RESERVE

    def get_grid_array(self):
        """Retorna array NxN com os estados das células."""
        N = self.params["grid_size"]
        arr = np.zeros((N, N), dtype=int)
        for (x, y), cell in self._cells.items():
            arr[y, x] = cell.state
        return arr

    def get_property_boundaries(self):
        """Retorna lista de retângulos (x0,y0,w,h) e info por propriedade."""
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

    def step(self):
        """Executa um passo (= 1 ano)."""
        self.year += 1
        self.annual_converted = 0
        self.annual_detected = 0
        self.annual_infractions = 0
        self.annual_embargoes = 0
        self.annual_acps = 0

        self.schedule.step()
        self.datacollector.collect(self)

        # Salvar snapshot
        self.history.append({
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
        })

    def run(self, n_years=None):
        """Roda a simulação completa."""
        if n_years is None:
            n_years = self.params["n_years"]
        # Coleta estado inicial
        self.datacollector.collect(self)
        self.history.append({
            "year": 0,
            "grid": self.get_grid_array().copy(),
            "pct_forest": self._pct_state(FOREST),
            "pct_pasture": self._pct_state(PASTURE),
            "pct_soy": self._pct_state(SOY),
            "pct_degraded": self._pct_state(DEGRADED_PASTURE),
            "pct_embargoed": self._pct_state(EMBARGOED),
            "pct_regenerating": self._pct_state(REGENERATING),
            "annual_converted": 0,
            "annual_infractions": 0,
            "annual_embargoes": 0,
            "annual_acps": 0,
            "VD_admin": self._mean_VD_admin(),
            "VD_acp": self._mean_VD_acp(),
            "VD_total": self._mean_VD_total(),
            "VE": self._mean_VE(),
            "C": self._mean_C(),
            "boundaries": self.get_property_boundaries(),
        })
        for _ in range(n_years):
            self.step()
        return self.datacollector.get_model_vars_dataframe()
