# Turbinis-Calc — Autonomous Multi-Architecture GPU Design Space Explorer

## Автономное итоговое техническое задание для OpenCode + GPT-6 Luna + DeepSeek

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

## 26. Benchmark — только достаточный для инженерного решения

Benchmark нужен не ради самого benchmark, а чтобы выбрать реально быстрый backend и размер batch.

Начать с минимального набора:

```text
1
100
1 000
10 000 candidates
```

для доступных путей:

```text
reference/original Python
optimized CPU
JAX CPU
JAX GPU
```

Далее увеличивать объём (`100 000`, `1 000 000` и выше) только если это необходимо для оценки throughput крупного sweep, выбора batch size, проверки scaling, RAM/VRAM/storage или доказательства практического выигрыша GPU.

Если уже на меньшем benchmark достоверно видно, какой backend использовать, не тратить время на низкоценные дополнительные размеры.

Отдельно измерять только то, что влияет на решение: JIT compile, warm execution, host↔device overhead, compute throughput и при необходимости Parquet throughput.

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

## 35. Главный режим работы — AUTONOMOUS UNTIL DONE

Это ТЗ выполняется не как серия этапов с подтверждениями пользователя, а как непрерывная автономная задача.

Главное правило:

```text
IF THE NEXT STEP FOLLOWS FROM THIS SPEC
AND DOES NOT REQUIRE NEW USER INFORMATION
→ DO IT NOW
```

Запрещено заканчивать работу фразами вроде «следующим шагом нужно...», «могу продолжить...», «подтвердите переход...» или «если хотите, я сделаю...», если следующий шаг уже однозначно следует из этого ТЗ.

После каждого этапа:

```text
IMPLEMENT
→ TEST
→ GPT-6 LUNA INTERNAL REVIEW
→ FIX IF NEEDED
→ UPDATE CHECKLIST / STATE
→ CONTINUE AUTOMATICALLY
```

Проверка GPT-6 Luna — внутренняя, а не запрос подтверждения пользователя.

Если review = FAIL:

1. записать причину;
2. поручить исправление подходящему агенту;
3. исправить;
4. повторить тесты;
5. повторить review;
6. после PASS автоматически продолжить.

Не останавливаться из-за обычных рабочих проблем: test failure, regression mismatch, Python/JAX exception, CUDA/JAX configuration, OOM, неподходящего batch size, solver convergence failure, dependency conflict, несовместимой функции с JAX, ошибки пути, сериализации, Parquet или precision. Их нужно диагностировать, исправлять и повторять проверку.

### 35.1. Настоящие BLOCKER'ы

Остановка допустима только если продолжение действительно невозможно без внешней информации или доступа:

- отсутствует необходимый исходный файл и его невозможно восстановить из проекта;
- требуется credential/API key, которого нет;
- нет доступа к репозиторию/файлу, без которого нельзя выполнить физическую модель;
- существует фундаментальное неоднозначное инженерное решение, для которого источники дают несколько несовместимых вариантов и выбор существенно меняет итог;
- необходимое внешнее действие физически невозможно выполнить из текущей среды.

Даже при BLOCKER:

```text
DO ALL INDEPENDENT WORK FIRST
```

Не блокировать весь проект задачей, которая мешает только одной ветке. Отметить её `[B]`, выполнить все независимые пункты и обращаться к пользователю только когда больше ничего полезного выполнить нельзя.

---

## 36. Правило скорости и отсечения ненужной работы

Цель — закончить инженерный проект, а не выполнить максимальное количество ритуальных задач.

Каждую новую задачу классифицировать:

```text
P0 — нужна для физической/численной корректности или итогового результата
P1 — непосредственно ускоряет расчёт или открывает необходимый следующий этап
P2 — полезное улучшение, но не требуется для Definition of Done
P3 — косметика / дублирование / nice-to-have
```

Порядок:

```text
P0 → P1 → P2 only if necessary
P3 → SKIP
```

Если задача не влияет на корректность физики, воспроизводимость, массовый sweep, существенный bottleneck, сравнение архитектур, анализ design space или итоговые инженерные выводы — не делать её без необходимости.

Пропуск отмечать:

```text
[S] SKIPPED — <краткая причина>
```

### 36.1. По умолчанию не делать

Без доказанной необходимости не тратить время на:

- переписывание всего проекта на JAX;
- C++/Rust/CUDA без profiling-доказательства;
- косметическую полировку CLI;
- несколько эквивалентных storage formats;
- дублирующие графики;
- повторный физический sweep при изменении только acceptance criteria;
- повторное чтение уже проверенных источников без причины;
- exhaustive benchmark всех размеров, если решение уже очевидно;
- оптимизацию холодных функций;
- native Windows GPU backend, если WSL2/JAX GPU закрывает задачу;
- дополнительные архитектуры вне ТЗ;
- огромный Cartesian product зависимых параметров;
- преждевременную микроптимизацию;
- документацию, не помогающую понять, запустить, проверить или продолжить проект.

### 36.2. Быстрый цикл разработки

```text
small smoke test
→ focused regression
→ representative batch
→ only then large run
```

Не запускать дорогой глобальный sweep после каждой правки.

Если изменился только analysis/constraint profile — повторно использовать RAW dataset.

Если изменился физический расчёт — пересчитать необходимые regression/control points, затем затронутые datasets.

Если изменился hot kernel без изменения физики — сравнить reference vs accelerated на репрезентативном наборе, не пересчитывая сразу всё пространство.

---

## 37. Постоянный CHECKLIST проекта

В репозитории должен существовать живой checklist.

Предпочтительный файл:

```text
docs/PROJECT_CHECKLIST.md
```

Если уже существует эквивалентный roadmap/status/tasks-файл — использовать его и не создавать дубль.

Статусы:

```text
[ ] TODO
[~] IN PROGRESS
[x] DONE
[B] BLOCKED
[S] SKIPPED
```

У каждого существенного выполненного пункта хранить короткое доказательство: файл, test name, run_id, численное сравнение или другой проверяемый результат.

Checklist — источник истины о прогрессе.

Перед новой работой:

1. прочитать checklist;
2. найти первый незавершённый P0/P1 пункт;
3. отметить `[~]`;
4. выполнить;
5. проверить;
6. отметить `[x]`, `[B]` или `[S]`;
7. сразу перейти к следующему.

Не держать состояние только в контексте чата.

### 37.1. MASTER CHECKLIST

Агент создаёт или обновляет `docs/PROJECT_CHECKLIST.md` по следующей структуре:

```markdown
# Turbinis-Calc Project Checklist

## M0 — Bootstrap / project state
- [ ] Найти существующие roadmap/status/docs и не создавать дубли.
- [ ] Создать/обновить PROJECT_CHECKLIST.
- [ ] Создать/обновить PROJECT_STATE.
- [ ] Создать/обновить PROJECT_GUIDE.
- [ ] Зафиксировать текущий git commit и рабочее состояние.
- [ ] Проверить shell/Python/tests для агентов.
- [ ] Зафиксировать environment и доступные CPU/GPU backends.

## M1 — Source audit + trustworthy reference
- [ ] Построить repository map.
- [ ] Найти основной notebook и computational call graph.
- [ ] Найти расчёты пользователя, Сергея и Антона.
- [ ] Найти учебник, scan/Markdown и связанные источники.
- [ ] Восстановить текущий gas path.
- [ ] Восстановить shaft layout.
- [ ] Однозначно определить смысл N=110 kW.
- [ ] Определить фактическое число compressor/turbine stages.
- [ ] Найти hardcoded N, Tг, πк, stage counts, weights, constraints.
- [ ] Найти early rejection и классифицировать A/B/C/D.
- [ ] Найти legacy/duplicate calculation paths.
- [ ] Найти source discrepancies.
- [ ] Выбрать reference backend.
- [ ] Зафиксировать минимум один контрольный расчёт текущего двигателя.
- [ ] Добавить regression tests ключевых промежуточных величин.
- [ ] Выполнить короткий baseline profile.
- [ ] Отметить hot functions и JAX compatibility.
- [ ] Не продолжать аудит ради аудита, если данных уже достаточно для M2.

## M2 — Unified physical core + architectures
- [ ] Выделить единый source of truth физических формул.
- [ ] Создать/нормализовать StationState.
- [ ] Реализовать переиспользуемые component models.
- [ ] Реализовать EngineArchitecture.
- [ ] Воспроизвести текущую одновальную схему.
- [ ] Реализовать single-shaft / single-CC.
- [ ] Реализовать single-shaft / dual-CC.
- [ ] Реализовать free-power-turbine architecture.
- [ ] Для каждой архитектуры обеспечить собственный cycle.
- [ ] Реализовать отдельные shaft power balances.
- [ ] Проверить mass balance.
- [ ] Проверить energy balance.
- [ ] Реализовать полный пересчёт cycle при изменении πк.
- [ ] Проверить compressor pressure-ratio distribution.
- [ ] Проверить turbine pressure-ratio distribution.
- [ ] Подтвердить/исправить stage-count assumptions.
- [ ] Reproduce reference points после рефакторинга.

## M3 — Full calculation semantics + RAW data
- [ ] Убрать engineering early rejection из calculation path.
- [ ] Оставить остановку только для mathematical/non-physical failure.
- [ ] Разделить calculation_status и constraint_status.
- [ ] Сохранять diagnostic record для failures.
- [ ] Создать RAW schema.
- [ ] Добавить run_id/candidate_id/architecture/sampling_method.
- [ ] Сохранять ключевые causal-chain intermediates.
- [ ] Реализовать constraint profiles поверх RAW.
- [ ] Реализовать margins/normalized violations/pass-fail.
- [ ] Сделать RAW неизменяемым по смыслу pipeline.
- [ ] Реализовать manifest.
- [ ] Реализовать reproduce(candidate).
- [ ] Проверить round-trip одного candidate.

## M4 — Performance path
- [ ] Измерить hot path.
- [ ] Удалить очевидные Python bottlenecks без изменения физики.
- [ ] Векторизовать только горячие kernels.
- [ ] Реализовать optimized CPU path там, где это выгодно.
- [ ] Реализовать JAX CPU для подходящих kernels.
- [ ] Сравнить reference vs JAX CPU.
- [ ] Настроить JAX GPU/WSL2 RTX 4060.
- [ ] Выполнить реальный numerical GPU self-test.
- [ ] Реализовать batch/chunk execution.
- [ ] Реализовать auto batch size / OOM recovery.
- [ ] Сравнить reference vs JAX GPU.
- [ ] Проверить float64.
- [ ] Проверять float32/mixed только если это реально ускоряет.
- [ ] Выполнить минимально достаточный benchmark.
- [ ] Выбрать production backend по измерениям.
- [ ] Не оптимизировать оставшиеся cold paths.

## M5 — Sweep engine
- [ ] Реализовать deterministic grid для ключевых outer variables.
- [ ] Реализовать constraint-preserving parameterization.
- [ ] Реализовать batch generation.
- [ ] Реализовать streaming Parquet write.
- [ ] Реализовать checkpoint/resume.
- [ ] Проверить resume после искусственного прерывания.
- [ ] Реализовать quick preset.
- [ ] Реализовать normal preset.
- [ ] Реализовать deep preset только если оправдан runtime/storage.
- [ ] Оценивать N_candidates/storage перед крупным run.
- [ ] Проверить все архитектуры на small sweep.
- [ ] Проверить πк=3...13 на coarse sweep.
- [ ] Проверить compressor split.
- [ ] Проверить turbine split.
- [ ] Проверить основные geometry/search variables.
- [ ] Не запускать бессмысленный Cartesian explosion.

## M6 — Production design-space calculation
- [ ] Запустить достаточный coarse RAW sweep всех архитектур.
- [ ] Проверить dataset integrity.
- [ ] Проверить numerical failure statistics.
- [ ] Проверить energy/mass residual distributions.
- [ ] Найти boundaries/high gradients/extrema.
- [ ] Выполнить adaptive refinement только там, где он добавляет информацию.
- [ ] Выполнить local perturbations вокруг важных candidates.
- [ ] При необходимости расширить sampling.
- [ ] Не считать дополнительные миллионы точек без ожидаемого информационного выигрыша.

## M7 — Engineering analysis
- [ ] Создать annotated dataset.
- [ ] Построить feasibility map.
- [ ] Построить near-feasibility analysis.
- [ ] Определить active constraints.
- [ ] Выполнить controlled sensitivity analysis.
- [ ] Исследовать влияние πк на G/πт/geometry/blade heights.
- [ ] Исследовать влияние n и D на blade heights/stress/angles.
- [ ] Исследовать compressor split.
- [ ] Исследовать turbine split.
- [ ] Сравнить архитектуры на одинаковой базе.
- [ ] Построить только информативные plots/heatmaps.
- [ ] Построить Pareto.
- [ ] Выбрать feasible/near-feasible/boundary finalists.
- [ ] Пересчитать finalists reference backend.
- [ ] Объяснить результаты физически, не только статистически.

## M8 — Final audit / delivery
- [ ] Reference tests PASS либо расхождения документированы.
- [ ] Energy balances проверены.
- [ ] Mass balances проверены.
- [ ] CPU fallback работает.
- [ ] GPU path реально работает на RTX 4060 либо blocker документирован.
- [ ] Checkpoint/resume проверен.
- [ ] Reproduce candidate проверен.
- [ ] RAW dataset сохранён.
- [ ] Analysis outputs сохранены.
- [ ] PROJECT_GUIDE соответствует фактическому проекту.
- [ ] PROJECT_STATE соответствует фактическому состоянию.
- [ ] Нет критических TODO/BLOCKED, делающих выводы недостоверными.
- [ ] Сформирован final engineering report.
- [ ] Даны ответы на основные инженерные вопросы ТЗ.
- [ ] Выполнен финальный GPT-6 Luna review.
```

---

## 38. PROJECT_STATE — текущее состояние работы

Вести:

```text
docs/PROJECT_STATE.md
```

или существующий эквивалент.

Это короткий оперативный файл, не длинный журнал.

Минимальная структура:

```markdown
# Project State

Last updated:
Git commit:
Current milestone:
Current task:
Current backend:
Current environment:

## Current status
Кратко 5–15 строк.

## Last verified results
- ...

## Current blockers
- none

## Important unresolved engineering questions
- ...

## Next executable task
- ...

## Latest datasets/runs
- run_id:
- path:
- config:
- status:

## Latest regression status
- ...

## Latest performance status
- ...
```

После каждого существенного блока обновлять файл. `Next executable task` должен позволять новому агенту продолжить работу без чтения истории чата.

---

## 39. PROJECT_GUIDE — как реально работает проект

Если такого файла ещё нет, создать:

```text
docs/PROJECT_GUIDE.md
```

Если уже существует `ARCHITECTURE.md`, `README` или другой документ, выполняющий эту функцию, обновлять его вместо создания дублирующего документа.

Этот файл описывает фактическую реализацию, а не первоначальный план. Обновлять его по мере изменений.

Минимальное содержание:

```markdown
# How Turbinis-Calc Works

## 1. Purpose
Что рассчитывает проект и какие design points исследует.

## 2. Engine architectures
Фактически реализованные схемы, gas paths и shafts.

## 3. Calculation flow
input
→ cycle
→ compressor
→ combustor
→ turbine
→ geometry
→ strength
→ RAW result
→ constraints
→ analysis

## 4. Source of truth
Какие модули основные и какие legacy.

## 5. Physical models
Основные формулы/методики и источники.

## 6. State / station definitions
Определения T*, p*, G, πк, πт, N и station numbering.

## 7. Power and mass balances
Как считаются shafts и cooling/bleed/fuel flows.

## 8. Backends
reference / NumPy / JAX CPU / JAX GPU.

## 9. Search space
Что независимо, что вычисляется.

## 10. Dataset pipeline
RAW → annotated → analysis → Pareto/finalists.

## 11. Constraints
Что относится к физике, что к acceptance criteria.

## 12. Commands
Как выполнить check, test, calculate, sweep, analyze, reproduce.

## 13. Environment
Windows / WSL2 / Linux, зависимости, GPU.

## 14. Results layout
Где manifests, raw data, plots, reports, checkpoints.

## 15. How to resume work
Какие файлы читать и какую команду запускать.

## 16. Known limitations
Отсутствующая физика, допущения, неподтверждённые места.
```

Если реализация меняется — PROJECT_GUIDE меняется в том же milestone.

---

## 40. Внутренний review вместо остановки

После каждого milestone GPT-6 Luna выполняет короткий internal gate:

```text
PHYSICS: PASS / FAIL
NUMERICS: PASS / FAIL
REGRESSION: PASS / FAIL
DATA INTEGRITY: PASS / FAIL
PERFORMANCE: PASS / FAIL / N/A
DOCUMENTATION: PASS / FAIL
NEXT MILESTONE SAFE: YES / NO
```

Результат кратко записывается в `PROJECT_STATE.md`.

Если `NEXT MILESTONE SAFE = NO`:

```text
find cause
→ fix
→ test
→ review again
```

Не обращаться автоматически к пользователю.

---

## 41. Git и безопасность автономной работы

Разрешено автономно:

- читать/создавать/изменять файлы проекта;
- запускать Python/tests/Jupyter;
- запускать JAX/GPU diagnostics;
- профилировать;
- создавать datasets;
- выполнять безопасные git status/diff/log;
- делать локальные commits после стабильных milestones, если этот workflow уже используется.

Не выполнять без явного разрешения пользователя:

- `git push --force`;
- destructive reset чужих изменений;
- удаление данных вне проекта;
- изменение credentials;
- публикацию секретов;
- потенциально разрушительные системные изменения Windows/WSL;
- удаление исходных reference calculations.

Перед крупным рефакторингом сохранять воспроизводимый reference.

---

## 42. Политика тестов для ускорения

Не запускать весь test suite после каждого изменения.

```text
edit
→ targeted test
→ affected regression tests
→ milestone test suite
```

Full regression нужен после изменения физических формул, cycle, architecture coupling, units/conventions, перед production sweep и перед финальным отчётом.

Для чистого refactor/performance change сначала targeted regression; полный suite — на milestone gate.

---

## 43. Политика больших расчётов

Перед большим run:

1. smoke test;
2. small representative batch;
3. correctness;
4. checkpoint/resume;
5. storage path;
6. GPU utilization;
7. оценка N candidates;
8. оценка storage;
9. production run.

Если run слишком дорог:

- сначала уменьшить resolution, но не физическое пространство;
- использовать hierarchical grid;
- refinement;
- resumable partitions;
- сохранить coarse coverage.

Не вырезать области только потому, что они нарушают engineering constraints.

---

## 44. Минимальный набор конечных результатов

Обязательны:

```text
working physical core
reference/regression tests
three required architecture models
RAW design-space dataset
constraint annotation
checkpoint/resume
CPU fallback
working accelerated path if technically viable
benchmark sufficient to justify backend choice
architecture comparison
sensitivity/feasibility/Pareto analysis
finalists
final engineering report
PROJECT_CHECKLIST
PROJECT_STATE
PROJECT_GUIDE
```

Все дополнительные инструменты, графики, команды, refactors и исследования выполнять только если они помогают получить или проверить эти результаты.

---

## 45. Definition of Done

### Physics
- архитектуры имеют согласованные cycles;
- shaft balances корректны;
- mass/energy balances проверены;
- `N`, `πк`, `πт`, `G`, stations однозначны;
- reference calculations воспроизводятся или расхождения объяснены.

### Calculation
- инженерно плохие, но вычислимые точки сохраняются;
- numerical/non-physical failures диагностируются;
- acceptance criteria отделены от calculation layer;
- `πк=3...13` приводит к полному пересчёту зависимых величин.

### Search
- исследованы требуемые архитектуры;
- compressor/turbine distributions;
- основные geometry variables;
- покрытие достаточно для инженерных выводов;
- refinement применён там, где coarse grid недостаточен.

### Performance
- production path измерен;
- CPU fallback работает;
- RTX 4060/JAX GPU используется для подходящего workload, если среда позволяет;
- accelerated backend подтверждён regression tests;
- крупные расчёты resumable.

### Data
- RAW dataset существует;
- manifest существует;
- datasets воспроизводимы;
- candidate можно пересчитать;
- constraint profiles применяются поверх RAW.

### Analysis
- найдены feasible/near-feasible regions;
- определены active constraints;
- выполнена sensitivity analysis;
- выполнено сравнение архитектур;
- построен Pareto;
- finalists перепроверены reference backend.

### Documentation
- checklist актуален;
- state актуален;
- PROJECT_GUIDE описывает реальный проект;
- final engineering report готов.

### Final review
- GPT-6 Luna выполнил итоговый review;
- нет скрытого критического blocker;
- оставшиеся `[S]` действительно необязательны;
- оставшиеся `[B]`, если есть, явно ограничивают только документированные части результата.

---

## 46. Финальный инженерный отчёт

Не писать огромный дневник разработки. Ответить прежде всего:

1. Какая архитектура наиболее перспективна и почему?
2. Какой диапазон `πк` предпочтителен?
3. Нужны ли два ЦБК?
4. Даёт ли free power turbine преимущество?
5. Какие turbine stage pressure ratios получаются?
6. Какие blade heights достижимы?
7. Какие ограничения активны?
8. Что мешает текущей конструкции, если feasible solution нет?
9. Какие параметры эффективнее двигают конструкцию к feasible region?
10. Как меняются stress, angles, reaction, Mach и efficiency?
11. Насколько устойчивы выводы?
12. Какие допущения или отсутствующая физика ограничивают достоверность?
13. Какие finalists рекомендуются дальше?

Включать только необходимые доказательства, plots и tables.

---

## 47. Восстановление после прерывания

Один процесс/контекст OpenCode может быть внешне прерван, поэтому проект должен быть resumable by design.

В начале каждого запуска:

```text
1. read docs/PROJECT_CHECKLIST.md
2. read docs/PROJECT_STATE.md
3. read docs/PROJECT_GUIDE.md if needed
4. inspect git status
5. inspect latest run/checkpoint if task is computational
6. continue first [~], then highest-priority [ ] item
```

Не начинать аудит с нуля, если он завершён. Не спрашивать пользователя «на чём мы остановились», если это определяется из файлов состояния.

---

## 48. Старт выполнения

Начать немедленно с:

```text
BOOTSTRAP AUTONOMOUS STATE
```

1. проверить существующие эквиваленты `PROJECT_CHECKLIST`, `PROJECT_STATE`, `PROJECT_GUIDE`;
2. создать недостающие или обновить существующие;
3. перенести MASTER CHECKLIST;
4. отметить уже выполненные пункты только после проверки репозитория;
5. начать `M1 — Source audit + trustworthy reference`;
6. после M1 выполнить internal GPT-6 Luna review;
7. не останавливаться;
8. автоматически перейти к M2;
9. продолжать по checklist до Definition of Done.

Не ждать подтверждения между milestones.

Не заканчивать работу только потому, что milestone завершён.

Не выполнять P2/P3-задачи, если они не нужны для обязательного результата.

После выполнения необходимого — финальный audit и final engineering report.

Главный принцип:

```text
UNDERSTAND
→ IMPLEMENT
→ TEST
→ REVIEW
→ RECORD STATE
→ CONTINUE
→ FINISH THE PROJECT
```
