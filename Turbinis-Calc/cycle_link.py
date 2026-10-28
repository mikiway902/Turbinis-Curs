"""Связь Turbinis-Calc с Turbinis-Cycle.

Код-ячейки блокнота цикла исполняются подряд в отдельном словаре-namespace
(без графиков, имена не пересекаются с Calc). Сам блокнот цикла не меняется,
поэтому любая правка в нём подхватывается при следующем запуске Calc.
"""

import contextlib
import io
import json
import os
import warnings
from pathlib import Path
from typing import Any

ЦИКЛ_ПО_УМОЛЧАНИЮ = Path(
    os.environ.get("TURBINIS_CYCLE")  # переопределение пути (например, для тестов)
    or Path(__file__).resolve().parent.parent / "Turbinis-Cycle" / "Turbinis-Cycle.ipynb"
)


def load_cycle(path=ЦИКЛ_ПО_УМОЛЧАНИЮ, quiet=True) -> dict[str, Any]:
    """Выполнить блокнот цикла и вернуть его namespace (dict).

    quiet=True гасит print цикла (таблицы, предупреждения matplotlib).
    """
    os.environ.setdefault("MPLBACKEND", "Agg")  # без окон/графиков
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    ns: dict[str, Any] = {"__name__": "turbinis_cycle"}
    буфер = io.StringIO()
    plt = None
    прежний_бэкенд = None
    try:  # в Jupyter активен inline-бэкенд: на время расчёта цикла отключаем вывод графиков
        import matplotlib
        import matplotlib.pyplot as plt

        прежний_бэкенд = matplotlib.get_backend()
        plt.switch_backend("Agg")
    except ImportError:
        pass
    try:
        with contextlib.redirect_stdout(буфер), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
            for i, ячейка in enumerate(nb["cells"]):
                if ячейка["cell_type"] != "code":
                    continue
                # строки-магии IPython (%..., !...) в обычном exec не работают
                код = "".join(
                    строка
                    for строка in ячейка["source"]
                    if not строка.lstrip().startswith(("%", "!"))
                )
                try:
                    exec(compile(код, f"{Path(path).name}[ячейка {i}]", "exec"), ns)
                except Exception as e:
                    raise RuntimeError(
                        f"Ошибка в ячейке {i} блокнота цикла: {e!r}"
                    ) from e
    finally:
        if plt is not None:
            plt.close("all")
            if прежний_бэкенд:
                plt.switch_backend(прежний_бэкенд)
    if not quiet:
        print(буфер.getvalue())
    return ns
