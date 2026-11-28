# Turbinis-Calc — Multi-Architecture GPU Design Space Explorer

## Итоговое техническое задание для OpenCode + GPT-6 Luna + DeepSeek

> Это единое итоговое ТЗ. Оно заменяет предыдущие версии и дополнения.

## 1. Цель

Превратить существующий `Turbinis-Calc` в воспроизводимый инженерный комплекс для массового исследования пространства конструкций турбовального/турбовинтового двигателя.

Основной design point:

- полезная мощность `N = 110 кВт`;
- температура газа перед первой турбиной `Tг* = 1250 K`.

Необходимо исследовать разные архитектуры двигателя, `πк*`, распределение `πк` и `πт`, обороты, диаметры, высоты лопаток, реактивность, `u/c0`, углы, скорости, Mach, прочность, КПД, геометрию и остальные независимые параметры существующей методики.

Главный принцип:

```text
CALCULATE FIRST
STORE EVERYTHING
CLASSIFY SECOND
ANALYZE THIRD
OPTIMIZE LAST
```

Не подгонять расчёт под желаемый результат. Главный первичный результат массового расчёта — `RAW DESIGN SPACE DATASET`, а не только `BEST ENGINE`.

## 2. Роли агентов

### GPT-6 Luna

Главный технический руководитель и reviewer. Отвечает за физику, методику, архитектуру модели, декомпозицию, проверку формул и размерностей, энергетические и массовые балансы, regression tests, принятие инженерных решений и интерпретацию результатов.

GPT-6 Luna не должен автоматически принимать код или выводы DeepSeek.

### DeepSeek

Coding/research subagent. Использовать для анализа репозитория, поиска связей, рефакторинга, реализации модулей, тестов, профилирования, NumPy/JAX, GPU batch engine, хранения данных, CLI, визуализации и анализа datasets.

DeepSeek не имеет права самостоятельно менять инженерную методику без проверки GPT-6 Luna.

## 3. Основная машина

Разработка и основные расчёты выполняются на стационарном ПК:

```text
Host OS: Windows
CPU: AMD Ryzen 5 9600X
RAM: 32 GB DDR5-6400
GPU: NVIDIA RTX 4060 8 GB
Storage: NVMe SSD
```

Основной рекомендуемый GPU compute environment:

```text
Windows → WSL2 → Ubuntu → Python → JAX → CUDA → RTX 4060
```

Перед установкой проверить актуальный официально поддерживаемый способ установки JAX/CUDA в WSL2. Не фиксировать версии CUDA/JAX по памяти.

Физическое ядро должно оставаться platform-independent. Желательная поддержка: Windows host, WSL2+Ubuntu, native Linux/Fedora и CPU fallback.

Не хардкодить `C:\Users\...`, `/home/...` и другие пользовательские абсолютные пути. Использовать `pathlib.Path`, project-relative paths и конфигурацию.

## 4. Источники

До существенных изменений найти и изучить:

1. текущий расчёт пользователя;
2. основной Jupyter notebook;
3. существующие Python-модули;
4. текущий optimizer;
5. расчёт Сергея;
6. расчёт Антона;
7. учебник;
8. оригинальный скан учебника;
9. Markdown/распознанную версию учебника;
10. существующие графики;
11. предыдущие результаты оптимизации;
12. документы/ТЗ;
13. экспериментальные реализации;
14. legacy-пути расчёта.

Не предполагать имена файлов — искать по содержимому репозитория.

Сопоставить:

```text
textbook scan ↔ textbook Markdown ↔ user calculation
↔ Sergey calculation ↔ Anton calculation ↔ current code
```

Ни один источник не считать автоматически правильным.

При расхождении создавать `SOURCE_DISCREPANCY` с формулами, источниками, численной разницей и возможной причиной. Не выбирать молча.

Markdown использовать для поиска, но подозрительные формулы сверять с оригинальным сканом, особенно индексы, степени, дроби, знаки, греческие буквы и градусы/радианы.

## 5. Сначала восстановить существующую модель

До рефакторинга установить:

- фактический gas path;
- количество компрессоров и компрессорных ступеней;
- количество турбинных агрегатов и турбинных ступеней;
- количество валов;
- механические связи;
- расположение редуктора и винта;
- точное определение `N=110 кВт`;
- определения `πк`, `πт`, `G`;
- систему станций;
- cooling;
- losses;
- strength model.

Не полагаться только на текстовые описания — проверить код и исходные расчёты.

Особенно определить, что означает `110 кВт`: мощность турбины, вала, до/после редуктора, на винте или другое. Все архитектуры сравнивать по одному определению полезной мощности.

## 6. Исследуемые архитектуры

### A — одновальная, два ЦБК

Предварительно:

```text
INLET → ЦБК-1 → ЦБК-2 → COMBUSTOR → TURBINE → EXHAUST
```

Механически:

```text
ЦБК-1 + ЦБК-2 + TURBINE = ONE SHAFT → GEARBOX → PROPELLER
```

Сначала подтвердить фактическую структуру проекта.

### B — одновальная, один ЦБК

```text
INLET → ЦБК → COMBUSTOR → TURBINE → EXHAUST
ЦБК + TURBINE = ONE SHAFT → GEARBOX → PROPELLER
```

### C — двухвальная со свободной силовой турбиной

По расчёту Сергея, если источник подтверждает эту схему:

```text
INLET → ЦБК → COMBUSTOR → GG/COMPRESSOR TURBINE
      → FREE POWER TURBINE → EXHAUST
```

Механически:

```text
SHAFT 1: COMPRESSOR ↔ GG TURBINE
SHAFT 2: FREE POWER TURBINE → GEARBOX → PROPELLER
```

Двухвальная схема должна иметь собственный цикл. Нельзя просто разделить существующую работу/`πт` одновальной турбины.

Баланс вала 1:

```text
GG turbine power ≈ compressor power + mechanical losses
```

Баланс вала 2:

```text
free power turbine power ≈ gearbox input + shaft losses
```

с учётом точного определения полезных 110 кВт.

Свободная турбина получает реальное состояние после GG turbine (`T*`, `p*`, `G`, composition, cooling flows и т.д.), а не автоматически `Tг*=1250 K`.

## 7. Архитектура определяет цикл

Запрещено:

```text
ONE CYCLE → A/B/C
```

Нужно:

```text
DESIGN POINT
 ├─ ARCH A → CYCLE A → COMPONENT MODELS
 ├─ ARCH B → CYCLE B → COMPONENT MODELS
 └─ ARCH C → CYCLE C → COMPONENT MODELS
```

При этом не копировать формулы три раза. Создать переиспользуемые компоненты:

`inlet`, `compressor_stage`, `compressor`, `combustor`, `turbine_stage`, `turbine`, `shaft`, `gearbox`, `propeller_load`, `exhaust`.

Создать явный `EngineArchitecture`, например:

```text
SINGLE_SHAFT_SINGLE_CC
SINGLE_SHAFT_DUAL_CC
FREE_POWER_TURBINE
```

Рассмотреть `StationState` с минимум `T_total`, `p_total`, `mass_flow`, `cp`, `k` и другими необходимыми величинами.

## 8. Перебор πк*

Основная внешняя переменная:

```text
3 ≤ πк* ≤ 13
```

Для каждого `πк*` полностью пересчитывать цикл. Нельзя менять `πк`, оставляя `G`, compressor work, turbine requirements и другие зависимые параметры от старой точки.

Для одного ЦБК исходно исследовать примерно `πк=3...7.5`.

Для двух ЦБК исходно примерно `πк=7.5...13`.

`7.5` — конфигурируемая исходная проектная граница, а не фундаментальный предел. Около `πк≈7...8` по возможности считать обе архитектуры.

Для двух ЦБК перебирать `πк1/πк2`. Равное логарифмическое распределение — reference/start point, а не единственный вариант.

Проверять физическую реализуемость ЦБК по доступной методике: work, efficiency, outlet temperature, speed, diameter, peripheral speed, Mach, corrected flow, diffuser/mechanical constraints и т.д. Если физики не хватает — не придумывать, а выдавать `MISSING_COMPRESSOR_PHYSICS`.

## 9. Турбина и ступени

Сначала установить, что в проекте называется «турбиной», а что «турбинной ступенью».

Не переносить автоматически старую трёхступенчатую модель на все архитектуры. По проекту, Сергею, Антону и учебнику определить оправданное число ступеней для single-shaft turbine, GG turbine и free power turbine. Если методика допускает варианты — число ступеней может быть дискретной переменной в обоснованном диапазоне.

Желаемый проектный диапазон:

```text
1.8 ≤ πт,i ≤ 2.4
```

Но это acceptance criterion, а не фильтр расчётного движка.

Кроме `πт` проверять реальную загрузку через `u/c0`, `ΔL`, `ρ`, углы, peripheral speed, Mach, efficiency, geometry и strength.

Для многоступенчатой турбины исследовать распределение `πт1, πт2, ...`, не считая равномерное автоматически оптимальным.

Если подтверждено:

```text
πт,total = ∏ πт,i
```

использовать constraint-preserving параметризацию, например логарифмическую, чтобы не генерировать математически несовместимые наборы.

## 10. Фундаментальное правило: не отбрасывать плохие точки

Если точка физически и математически рассчиталась:

```text
STORE IT
```

Независимо от нарушения:

- blade height;
- `α`;
- `ρ`;
- `u/c0`;
- stress;
- Mach;
- `πт`;
- efficiency;
- compressor recommendations;
- geometry/manufacturing recommendations;
- других acceptance constraints.

Пример:

```text
l1 = 10.8 mm
l1_min = 14 mm
l1_margin = -3.2 mm
l1_ok = false
```

Точка сохраняется полностью.

Аналогично:

```text
sigma = 347 MPa
sigma_max = 300 MPa
margin = -47 MPa
strength_ok = false
```

или `α2`, `ρ`, Mach, `πт` вне желаемого диапазона.

Инженерно плохая точка — полезный результат, потому что показывает форму пространства и направление изменения конструкции.

## 11. Calculation failure ≠ engineering failure

Разделить:

```text
calculation_status
constraint_status
```

`SUCCESS + FAILED_CONSTRAINTS` — нормальный результат.

Если solver не сошёлся, появился NaN, division by zero, invalid root или undefined geometry — всё равно сохранить входы:

```text
calculation_status = NUMERICAL_FAILURE
failure_stage = ...
failure_reason = ...
```

Если состояние физически бессмысленно (`T_abs<=0`, `p_abs<=0` и т.п.), не продолжать искусственно, но сохранить:

```text
calculation_status = NON_PHYSICAL_STATE
failure_stage = ...
failure_reason = ...
last_valid_state = ...
```

## 12. Найти и убрать early rejection по acceptance constraints

Найти `return None`, `continue`, `break`, `reject` и аналоги внутри расчётного пути.

Классифицировать:

```text
A — математически невозможно продолжить
B — физически бессмысленно продолжить
C — нарушено только инженерное требование
D — legacy/unclear
```

Категория C не должна прекращать расчёт.

A/B создают diagnostic record.

D сначала исследовать.

## 13. RAW dataset

Главный результат sweep — `RAW DESIGN SPACE DATASET`.

Для каждой точки хранить:

### Identification
`run_id`, `candidate_id`, `architecture`, `sampling_method`.

### Inputs
`πк,total`, `πк1`, `πк2`, `Tг`, `N_required`, speeds, geometry, reaction, pressure-ratio distributions и все независимые search variables.

### Cycle
Station temperatures/pressures, mass flows, compressor/turbine work, fuel, cooling flows, cycle efficiency, energy/mass residuals.

### Compressor
Для каждого компрессора/ступени: pressure ratio, efficiency, temperatures, work, speed, diameter, peripheral velocity, Mach и другие доступные параметры.

### Turbine
Для каждой турбины/ступени: `πт`, work, efficiency, `c0`, `u/c0`, `ρ`, `α1`, `α2`, velocities, Mach, diameters, blade heights, flow areas, losses.

### Strength
Stresses, limits, margins.

### Mechanical
Shaft powers/speeds, gearbox ratio/efficiency, useful output.

### Diagnostics
`calculation_status`, `failure_stage`, `failure_reason`.

Сохранять промежуточные величины, чтобы можно было исследовать причинную цепочку:

```text
πк → compressor state/work → combustor → G
→ turbine work/πт → c0 → area → D/l
→ α/ρ/u-c0 → stress → efficiency
```

## 14. RAW и acceptance criteria разделены

Изменение, например:

```text
l_min: 14 → 20 mm
```

не должно требовать полного повторного физического sweep, если `l_min` не входит в сами физические уравнения.

Создать constraint profiles, например `BOOK`, `CURRENT`, `RELAXED`, `L14`, `L20`, `CUSTOM`, восстановив реальные существующие сценарии.

Поверх RAW создавать annotated dataset с:

```text
actual
min
max
lower_margin
upper_margin
normalized_violation
pass/fail
```

`feasible` — производное поле, а не условие сохранения строки.

Для near-feasible сохранять полный вектор нарушений, а не только один искусственный score.

## 15. Пространство поиска

Перебирать реальные независимые степени свободы:

```text
architecture
πк,total
compressor pressure-ratio distribution
turbine pressure-ratio distribution
rotational speeds
diameters
reaction
velocity ratios
geometry
other actual optimizer variables
```

при фиксированных `N=110 кВт`, `Tг*=1250 K`.

Не создавать бессмысленный Cartesian product математически зависимых величин.

## 16. Sampling

Для ключевых переменных использовать deterministic regular grid, чтобы строить чистые зависимости.

Использовать hierarchical refinement:

```text
coarse global grid
→ raw dataset
→ detect boundaries/high gradients/extrema
→ finer local grids
→ append data
```

Refinement добавляет данные и не удаляет coarse points.

Sobol/LHS допустимы как дополнение для высокой размерности, но не должны быть единственным доказательством отсутствия feasible region.

Для каждой точки хранить provenance:

```text
GRID
REFINEMENT
SOBOL
LHS
LOCAL_PERTURBATION
OPTIMIZER
```

## 17. Реальное GPU-ускорение

Массовый перебор должен быть специально спроектирован для RTX 4060.

Недостаточно `CUDA available=True`.

Основной кандидат:

```text
JAX + CUDA
```

Использовать по необходимости `jax.numpy`, `jit`, `vmap`, `lax`, `scan`.

Не переписывать проект сразу на JAX.

Порядок:

```text
reference Python
→ profiling
→ pure numerical functions
→ NumPy/vectorized CPU
→ JAX CPU
→ regression validation
→ JAX GPU
```

Correctness важнее процента кода на GPU.

## 18. Profiling

До оптимизации измерить реальные bottlenecks:

- cumulative time;
- call counts;
- Python loops;
- scalar calculations;
- SciPy/root solvers;
- repeated allocations/conversions;
- dictionaries/objects в hot path;
- IO.

Не оптимизировать наугад.

Горячее ядро по возможности привести к:

```text
arrays in → numerical operations → arrays out
```

## 19. GPU batching

Не отправлять по одной точке на GPU.

Нужно:

```text
large candidate batch → GPU → result arrays
```

Если функция считает одну точку — рассмотреть `vmap`; стабильные batch kernels — `jit`.

Отдельно измерять compile time и warm execution.

Разные архитектуры могут иметь отдельные kernels. Группировать batches по архитектуре и числу ступеней для стабильных shapes и уменьшения JAX recompilation.

## 20. CPU + GPU hybrid

Допустимо:

### CPU
generation, orchestration, configuration, IO, Parquet, irregular algorithms.

### GPU
cycle numerical kernels, stage calculations, geometry, losses, strength, large vector operations.

Решать по benchmark.

SciPy solvers не считать автоматически GPU-compatible. Рассматривать analytical/vectorized/JAX-compatible replacement или CPU fallback только при сохранении математической задачи.

## 21. Точность

Сначала валидировать `float64`.

Сравнить `float64`/`float32` для `N`, `G`, `T`, `p`, `η`, `π`, `α`, `ρ`, `l`, `D`, `σ`.

Mixed precision разрешать только после доказанной точности. Finalists пересчитывать reference precision.

## 22. Device selection и self-test

Поддержать:

```text
--device auto
--device cpu
--device gpu
--batch-size auto
```

`turbinis check` должен показать platform, Python/JAX, devices, CUDA backend, GPU name, precision, RAM/storage и выполнить реальный GPU numerical kernel.

Для 8 GB VRAM использовать chunking и автоматически подбирать разумный batch size без OOM.

## 23. Streaming storage

Pipeline:

```text
generate batch
→ GPU calculation
→ transfer result arrays
→ append Parquet
→ checkpoint
→ release memory
→ next batch
```

Не держать миллионы точек в `list[dict]` или одновременно в RAM/VRAM.

Предпочтительно Parquet/Arrow/Polars и partitioned dataset:

```text
results/raw/architecture=.../run=.../part-*.parquet
```

Для WSL2 benchmark Linux filesystem против `/mnt/c/...`; тяжёлый compute/data path держать там, где измеренно быстрее. Финальные datasets можно экспортировать в Windows.

## 24. Checkpoint, manifest, reproducibility

Каждый batch безопасно записывать. `resume` продолжает прерванный sweep.

Run manifest хранит:

```text
run_id
git_commit
configuration
model_version
backend
precision
device
GPU/CPU model
batch_size
parameter ranges
grid spacing
sampling methods
timestamp
```

Команда:

```bash
turbinis reproduce <run_id> <candidate_id>
```

должна повторять конкретную точку.

## 25. Reference и regression

Сохранить исходный подтверждённый backend.

Сравнивать:

```text
old/reference
vs new CPU
vs JAX CPU
vs JAX GPU
```

Проверять не только N, но station T/p, G, compressor/turbine work, `πт`, `c0`, `u/c0`, `ρ`, `α`, `D`, `l`, stress, efficiency.

Не увеличивать tolerance только ради зелёного теста.

Reference points: текущий двигатель пользователя, Сергей, Антон и существующие проверенные optimizer points.

Если Сергей действительно описывает free-power architecture — сначала воспроизвести его контрольную точку.

## 26. Benchmark

Измерить по возможности:

```text
1
100
1 000
10 000
100 000
1 000 000 candidates
```

для:

```text
original Python
optimized CPU
JAX CPU
JAX GPU
```

Отдельно: JIT compile, warm execution, host↔device overhead, compute throughput, Parquet throughput.

Не задавать заранее выдуманный коэффициент ускорения.

## 27. Presets и размер sweep

После profiling/benchmark определить:

```text
quick
normal
deep
exhaustive
```

Начальная идея для `πк`: `3...13`, например шаг `0.1`, но окончательный шаг выбрать после оценки sensitivity, runtime и storage.

До крупного run показывать:

```text
N_candidates
estimated raw/annotated size
estimated CPU runtime
measured/estimated GPU runtime
estimated storage
```

Если пространство слишком велико, разрешены staged grids, изменение resolution, mathematically valid parameterization, разделение runs, Sobol/LHS и refinement. Нельзя сокращать пространство просто из-за нарушения acceptance constraints.

## 28. Analysis pipeline

Команда:

```bash
turbinis analyze <run>
```

должна создавать annotation, feasibility, near-feasibility, sensitivity и Pareto views.

Исследовать минимум:

```text
∂l/∂πк
∂l/∂n
∂l/∂D
∂α/∂πк
∂α/∂ρ
∂σ/∂n
∂σ/∂D
∂η/∂πк
∂η/∂πт
```

где это корректно.

Поддержать controlled slices и local perturbations (`x±1%`, `x±2%` или обоснованные диапазоны).

Не путать корреляцию с причинностью: использовать controlled slices, finite differences и физические уравнения.

Система должна отвечать на обратные вопросы: что менять для роста `l1`, попадания `α2` в диапазон, снижения `σ`, улучшения `πт` и т.п.

## 29. Визуализация

Минимальные зависимости:

```text
πк → G
πк → η
πк → πт
πк → l
πк → α
πк → ρ
πк → σ
n → l
n → σ
D → l
D → α
πт → l
πт → η
```

Heatmaps/contours, например:

```text
πк × n → l1
πк × D → α2
n × D → σ
πк × compressor split → η
```

Накладывать contours `l=14 mm`, `σ=300 MPa`, `πт=1.8/2.4`, не скрывая точки вне них.

## 30. Честное сравнение архитектур

Сравнивать A/B/C при одинаковых полезной мощности, `Tг*`, ambient/inlet assumptions и единых определениях КПД, если физика схемы не требует иного.

Для одновальной схемы compressor/turbine на одном валу имеют общую частоту. Для двухвальной `n_compressor=n_GG_turbine`, а `n_power_turbine` может быть независимой переменной.

Учитывать gearbox ratio/efficiency.

Для каждой точки сохранять energy и mass balance residuals, включая cooling/bleed/fuel/return flows, если они есть в модели.

## 31. Ограничения и старые сценарии

Автоматически собрать все реальные constraints проекта: `α`, `ρ`, `u/c0`, `l`, `σ`, Mach, `n`, `D`, `πт`, geometry, compressor limits и другие.

Не считать заранее известный список полным.

Сохранить существующие сценарии вроде `BOOK`, `RELAXED`, `COMPROMISE`, `CONST_HUB`, если они реально существуют.

## 32. Pareto и finalists

Не выбирать лучший двигатель одним произвольным score.

Строить Pareto по релевантным показателям: efficiency, blade height, stress, size, rotational speed, constraint margins, architecture complexity и др.

После RAW + analysis выбрать best feasible, best near-feasible, representative alternatives и boundary cases. Каждый finalist пересчитать reference backend.

Итоговая таблица по каждой архитектуре должна содержать минимум:

```text
best feasible / near-feasible
πк
compressor split
G
turbine arrangement
πт по ступеням
shaft speeds
diameters
blade heights
α
ρ
u/c0
Mach
stress
efficiency
constraint margins
energy residual
mass residual
```

## 33. Вопросы, на которые должен ответить проект

1. Какая архитектура наиболее перспективна и почему?
2. При каком `πк`?
3. Как `πк` влияет на `G`, `πт`, геометрию и лопатки?
4. Какие параметры сильнее всего влияют на `α2`, stress и efficiency?
5. Почему текущая схема проходит/не проходит ограничения?
6. Что нужно изменить, чтобы приблизить её к допустимой области?
7. Даёт ли free power turbine преимущество?
8. Даёт ли независимая скорость power turbine преимущество?
9. Нужны ли два ЦБК?
10. Где разумная граница между одним и двумя ЦБК?
11. Какие `πт` реально получаются и где они попадают в `1.8...2.4`?
12. Какие решения устойчивее к изменению параметров?

## 34. CLI и установка

Целевые команды:

```bash
turbinis check
turbinis benchmark
turbinis calculate
turbinis sweep
turbinis analyze
turbinis compare
turbinis reproduce
```

Основной запуск:

```bash
turbinis sweep   --architectures all   --power 110   --t-g 1250   --pi-k 3:13   --preset deep   --device gpu
```

Подготовить `setup-wsl.sh`, `setup-linux.sh`, а `setup-windows.ps1` — если реально нужен. Общие зависимости держать в `pyproject.toml`/lock files.

Не ломать Jupyter. Вынести computational core в модули, а notebooks оставить для configure → calculate → analyze → visualize.

Стремиться к единому source of truth для физических формул и backends `reference`, `numpy/cpu`, `jax/cpu`, `jax/gpu`.

## 35. PHASE 0 — начать только с аудита

Сейчас начать только:

```text
PHASE 0 — SOURCE & REPOSITORY AUDIT
```

На этом этапе:

- не менять физические формулы;
- не переписывать проект;
- не начинать JAX/GPU migration;
- не запускать глобальный sweep/optimizer;
- не добавлять новые архитектуры.

Построить repository map и фактический call graph. Найти hardcoded `N=110`, `Tг=1250`, `πк=11`, stage counts, pressure-ratio weights, angle/blade/stress limits, cooling assumptions.

Восстановить gas path и shaft layout пользователя, Сергея и Антона. Сопоставить ключевые формулы с учебником.

Найти все early rejections и классифицировать A/B/C/D.

Выполнить baseline profiling без оптимизации.

Для каждой горячей функции определить:

```text
JAX READY
EASY REFACTOR
HARD REFACTOR
CPU ONLY
UNCLEAR
```

Оценить coarse/normal/deep candidate counts, storage и CPU runtime. GPU runtime до benchmark не выдумывать.

### Обязательные вопросы Phase 0

1. Что реально означает `N=110 кВт`?
2. Какова текущая архитектура?
3. Сколько реально ЦБК?
4. Сколько турбинных агрегатов и ступеней?
5. Как рассчитываются `πк`, `πт`, `G`?
6. Как связаны `N`, `G`, `πк`, `πт`?
7. Как меняется цикл при `πк`?
8. Можно ли корректно считать `πк=3...13`?
9. Какая модель ЦБК существует и чего ей не хватает?
10. Как устроены схемы Сергея и Антона?
11. Можно ли использовать их как regression references?
12. Какие формулы между источниками различаются?
13. Какие constraints преждевременно отбрасывают точки?
14. Какие функции занимают больше всего времени?
15. Что реально можно вынести на RTX 4060?
16. Что мешает batch execution/JAX?
17. Какие solvers используются?
18. Какие параметры действительно независимы?
19. Какие пути legacy/duplicate?
20. Какие риски единиц, градусов/радиан и размерностей существуют?

### Отчёт Phase 0

```text
REPOSITORY MAP
SOURCE FILES
USER ARCHITECTURE
SERGEY ARCHITECTURE
ANTON ARCHITECTURE
TEXTBOOK MAPPING
GAS PATHS
SHAFT LAYOUTS
CURRENT CYCLE MODEL
CURRENT COMPRESSOR MODEL
CURRENT TURBINE MODEL
CURRENT COOLING MODEL
CURRENT STRENGTH MODEL
CURRENT OPTIMIZER
CURRENT SEARCH SPACE
CURRENT CONSTRAINTS
EARLY REJECTIONS
SOURCE DISCREPANCIES
HARDCODED PARAMETERS
LEGACY / DUPLICATED CODE
CALL GRAPH
BASELINE PERFORMANCE
BOTTLENECKS
JAX COMPATIBILITY
GPU MIGRATION CANDIDATES
CPU-ONLY CANDIDATES
NUMERICAL RISKS
MISSING PHYSICS
ESTIMATED DATASET SIZE
RECOMMENDED ARCHITECTURE
PHASE 1 PLAN
```

После этого `STOP` и проверка GPT-6 Luna.

## 36. Дальнейшие фазы

```text
PHASE 1  — Reference Suite
PHASE 2  — Unified Physics Core
PHASE 3  — Unified Cycle Model
PHASE 4  — Reproduce Current Architecture
PHASE 5  — Single-Shaft / Single-CC
PHASE 6  — Free-Power-Turbine Architecture
PHASE 7  — Full-Calculation Mode without engineering rejection
PHASE 8  — RAW Data Schema / Manifest / Reproducibility
PHASE 9  — CPU Vectorization
PHASE 10 — JAX CPU
PHASE 11 — JAX CUDA / RTX 4060
PHASE 12 — Streaming GPU Sweep Engine
PHASE 13 — πк=3...13 Full-Cycle Sweep
PHASE 14 — Compressor Distribution Sweep
PHASE 15 — Turbine Distribution Sweep
PHASE 16 — Full Geometry Sweep
PHASE 17 — Deep RAW Sweep for all architectures
PHASE 18 — Adaptive Refinement
PHASE 19 — Constraint Annotation
PHASE 20 — Sensitivity Analysis
PHASE 21 — Feasibility / Near-Feasibility
PHASE 22 — Pareto Analysis
PHASE 23 — Finalists
PHASE 24 — Reference Recalculation
PHASE 25 — Final Engineering Report
```

После каждой фазы вернуть:

```text
WHAT WAS INVESTIGATED
SOURCES USED
WHAT WAS FOUND
WHAT CHANGED
FILES CHANGED
FORMULAS CHANGED
TEST RESULTS
REGRESSION DIFFERENCES
ENERGY BALANCE
MASS BALANCE
PERFORMANCE
GPU UTILIZATION
DATASET SIZE
ENGINEERING ISSUES
SOURCE DISCREPANCIES
RISKS
NEXT STEP
```

Если пункт неприменим — `N/A`. Не выдумывать.

Не переходить молча к следующей фазе, особенно после изменения формул, цикла, архитектуры, численного метода или обнаружения расхождения с reference.

## 37. Критерии готовности

Проект готов только если:

- каждая архитектура имеет собственный физически согласованный цикл;
- проверяются балансы каждого вала;
- reference calculations воспроизводятся;
- плохие, но вычислимые точки не теряются;
- numerical/non-physical failures сохраняются диагностически;
- RTX 4060 реально выполняет batch numerical workload;
- есть CPU fallback;
- есть измеренные benchmarks;
- каждую точку можно воспроизвести;
- можно анализировать всё исследованное пространство;
- acceptance criteria отделены от физики там, где это корректно;
- `πк` и остальные независимые параметры реально исследуются;
- три архитектуры сравниваются на общей базе;
- финальные выводы объясняются физикой, балансами и данными, а не подгонкой или одним score.

## 38. Финальный принцип

Нужен не просто optimizer, а:

```text
ENGINE DESIGN SPACE EXPLORER
```

который показывает:

- где хорошие конструкции;
- где плохие;
- насколько и почему они плохи;
- какие параметры являются ограничителями;
- куда нужно двигать конструкцию;
- как меняются зависимости;
- как отличаются архитектуры;
- где находятся границы допустимости;
- насколько чувствителен результат к каждому параметру.

Не уничтожать информацию только потому, что точка не удовлетворяет текущим требованиям.

## 39. Начать сейчас

Начать **только с PHASE 0**.

Сначала полностью понять существующий проект, расчёты пользователя, Сергея и Антона, учебник/Markdown, optimizer, ограничения, bottlenecks и потенциальную GPU-векторизацию.

После Phase 0 предоставить полный отчёт GPT-6 Luna и **остановиться для проверки**.

Только после подтверждения двигаться дальше по фазам, сохраняя физическую и численную эквивалентность относительно reference implementation.
