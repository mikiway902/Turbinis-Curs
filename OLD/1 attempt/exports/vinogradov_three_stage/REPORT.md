# Связанный расчёт Turbinis-Calc / Виноградов

Согласование коэффициентов: **False**.
Допустимая конфигурация: **False**.

No admissible configuration in this documented run; NOT a proof over all possible designs.

Исходные ограничения сохранены. Мощность 110кВт относится к входу редуктора — действующая плоскость Turbinis-Calc, не плоскость двухступенчатых ресурсных ноутбуков.

## Фактические итерации

| Итерация | N, Вт | n, об/мин | lСА1, мм | η* | σ предварительное, МПа |
|---|---:|---:|---:|---:|---:|
|0|109999.994|75658.385|9.585|0.901712|287.119|

## Причины недопустимости последней итерации

- {'row': 'SA1', 'name': 'radial pressure momentum', 'value': 0.08689692455892445, 'limit': 0.01, 'passed': False, 'classification': 'numerical', 'source': 'Accepted radial/equilibrium.py', 'units': '1', 'note': ''}
- {'row': 'SA1', 'name': 'general radial equilibrium', 'value': 0.08692667019904261, 'limit': 0.01, 'passed': False, 'classification': 'numerical', 'source': 'Accepted radial/equilibrium.py', 'units': '1', 'note': ''}
- {'row': 'SA1', 'name': 'construction/calculation failure', 'message': 'Subsonic Vinogradov recommendation not applicable to trans/supersonic section', 'details': {'Mach_in': 0.10854993596508933, 'Mach_out': 1.004157957281678}}
- {'row': 'RK1', 'section': 0, 'name': 'global throat not bracketed', 'value': 0.0008616153873785098, 'units': 'm'}
- {'row': 'RK1', 'section': 0, 'name': 'strength/creep', 'radius_m': np.float64(0.040283163203022054), 'area_m2': np.float64(9.517693020548971e-06), 'normal_force_N': np.float64(1628.0754002927015), 'gas_moment_N_m': [-0.022309878631436825, 0.02224108609960221], 'tension_Pa': np.float64(171057775.952391), 'normal_min_Pa': 165106066.48714754, 'normal_max_Pa': 179717711.33695665, 'temperature_K': 1122.0165171998185, 'yield_Pa': 599892472.4260249, 'lm_allowable_Pa': 175962135.82131562, 'lm_margin': 0.9791029193077158, 'required_lm_margin': 1.0, 'screening_passed': True, 'material_passed': True, 'creep_passed': False}
- {'row': 'RK1', 'name': 'vibration strength', 'status': 'NOT_CHECKED', 'reason': 'NOT_CHECKED: no IN738C temperature-specific SN data, damping or harmonic load amplitude; axial beam modes are screening only'}
- {'row': 'SA2', 'name': 'radial pressure momentum', 'value': 0.08782825793807587, 'limit': 0.01, 'passed': False, 'classification': 'numerical', 'source': 'Accepted radial/equilibrium.py', 'units': '1', 'note': ''}
- {'row': 'SA2', 'name': 'general radial equilibrium', 'value': 0.08788098294894937, 'limit': 0.01, 'passed': False, 'classification': 'numerical', 'source': 'Accepted radial/equilibrium.py', 'units': '1', 'note': ''}
- {'row': 'SA2', 'name': 'construction/calculation failure', 'message': 'Subsonic Vinogradov recommendation not applicable to trans/supersonic section', 'details': {'Mach_in': 0.1710695312859686, 'Mach_out': 1.0209219131255673}}
- {'row': 'RK2', 'name': 'integral mass', 'value': 0.017445695937048145, 'limit': 0.01, 'passed': False, 'classification': 'numerical', 'source': 'Annular continuity', 'units': '1', 'note': ''}
- {'row': 'RK2', 'section': 0, 'name': 'strength/creep', 'radius_m': np.float64(0.0383032562596962), 'area_m2': np.float64(1.7370702070510233e-05), 'normal_force_N': np.float64(5416.063896917138), 'gas_moment_N_m': [-0.03325412664955116, 0.05194883274920947], 'tension_Pa': np.float64(311793033.7491564), 'normal_min_Pa': 309733621.55422777, 'normal_max_Pa': 316153538.9880982, 'temperature_K': 964.1777956752823, 'yield_Pa': 865655719.8597593, 'lm_allowable_Pa': 535981391.71019363, 'lm_margin': 1.6953199177389913, 'required_lm_margin': 1.0, 'screening_passed': False, 'material_passed': True, 'creep_passed': True}
- {'row': 'RK2', 'name': 'vibration strength', 'status': 'NOT_CHECKED', 'reason': 'NOT_CHECKED: no IN738C temperature-specific SN data, damping or harmonic load amplitude; axial beam modes are screening only'}
- {'row': 'SA3', 'name': 'general radial equilibrium', 'value': 0.010037513707670353, 'limit': 0.01, 'passed': False, 'classification': 'numerical', 'source': 'Accepted radial/equilibrium.py', 'units': '1', 'note': ''}
- {'row': 'SA3', 'name': 'construction/calculation failure', 'message': 'Subsonic Vinogradov recommendation not applicable to trans/supersonic section', 'details': {'Mach_in': 0.18546122117104188, 'Mach_out': 1.058527720788094}}
- {'row': 'RK3', 'section': 0, 'name': 'local streamtube choke conflict', 'possible': False, 'required_flux': 232.75290338719006, 'critical_flux': 193.49617478850269, 'excess_fraction': 0.2028811610441199, 'mach': None}
- {'row': 'RK3', 'section': 0, 'name': 'strength/creep', 'radius_m': np.float64(0.03620628395491886), 'area_m2': np.float64(2.294287657753398e-05), 'normal_force_N': np.float64(7923.745419241281), 'gas_moment_N_m': [-0.04148254350435744, 0.06797107403990588], 'tension_Pa': np.float64(345368436.79838884), 'normal_min_Pa': 343526109.9547978, 'normal_max_Pa': 349539510.3873541, 'temperature_K': 811.2688218060141, 'yield_Pa': 917407355.1416725, 'lm_allowable_Pa': nan, 'lm_margin': nan, 'required_lm_margin': 1.0, 'screening_passed': False, 'material_passed': True, 'creep_passed': False}
- {'row': 'RK3', 'section': 1, 'name': 'strength/creep', 'radius_m': np.float64(0.04227681617878297), 'area_m2': np.float64(1.7537678360487457e-05), 'normal_force_N': np.float64(5494.4915954236685), 'gas_moment_N_m': [-0.023333865483239513, 0.04213682375542471], 'tension_Pa': np.float64(313296405.74335116), 'normal_min_Pa': 310029733.84898937, 'normal_max_Pa': 319838102.305338, 'temperature_K': 821.9805997164084, 'yield_Pa': 916701483.3046802, 'lm_allowable_Pa': nan, 'lm_margin': nan, 'required_lm_margin': 1.0, 'screening_passed': False, 'material_passed': True, 'creep_passed': False}
- {'row': 'RK3', 'section': 2, 'name': 'strength/creep', 'radius_m': np.float64(0.04834734840264709), 'area_m2': np.float64(1.204371688263102e-05), 'normal_force_N': np.float64(3449.0607930104443), 'gas_moment_N_m': [-0.010370583298062425, 0.020478807004827737], 'tension_Pa': np.float64(286378435.0481159), 'normal_min_Pa': 282439801.1607227, 'normal_max_Pa': 294660480.27307725, 'temperature_K': 833.8668148773802, 'yield_Pa': 915918219.8165137, 'lm_allowable_Pa': nan, 'lm_margin': nan, 'required_lm_margin': 1.0, 'screening_passed': True, 'material_passed': True, 'creep_passed': False}
- {'row': 'RK3', 'section': 3, 'name': 'strength/creep', 'radius_m': np.float64(0.054417880626511206), 'area_m2': np.float64(1.0085630831834102e-05), 'normal_force_N': np.float64(1701.2362134228), 'gas_moment_N_m': [-0.002592641349511815, 0.0055501844358751135], 'tension_Pa': np.float64(168679207.2592077), 'normal_min_Pa': 166932992.35510093, 'normal_max_Pa': 172289779.40039912, 'temperature_K': 851.545089709424, 'yield_Pa': 914753278.1574414, 'lm_allowable_Pa': nan, 'lm_margin': nan, 'required_lm_margin': 1.0, 'screening_passed': True, 'material_passed': True, 'creep_passed': False}
- {'row': 'RK3', 'section': 4, 'name': 'strength/creep', 'radius_m': np.float64(0.06048841285037532), 'area_m2': np.float64(9.127891923063156e-06), 'normal_force_N': np.float64(-0.0), 'gas_moment_N_m': [-0.0, 0.0], 'tension_Pa': np.float64(-0.0), 'normal_min_Pa': 0.0, 'normal_max_Pa': 0.0, 'temperature_K': 871.3119413498706, 'yield_Pa': 913450705.9798146, 'lm_allowable_Pa': nan, 'lm_margin': nan, 'required_lm_margin': 1.0, 'screening_passed': True, 'material_passed': True, 'creep_passed': False}
- {'row': 'RK3', 'name': 'vibration strength', 'status': 'NOT_CHECKED', 'reason': 'NOT_CHECKED: no IN738C temperature-specific SN data, damping or harmonic load amplitude; axial beam modes are screening only'}
- {'scope': 'existing Turbinis-Calc', 'name': 'l_с1 >= 14 мм (проект)', 'value': '9.58 мм', 'limit': '>= 14 мм', 'ok': False, 'kind': 'project', 'amount': 0.3153881446358183, 'note': ''}
- {'scope': 'existing Turbinis-Calc', 'name': 'required first SA height', 'value': 0.009584565975098543, 'limit': 0.014, 'units': 'm'}

## Проверки источника

Абсолютные и относительные ошибки: [CSV](controls/control_errors.csv), [JSON](controls/control_errors.json).
![Контрольные построения](controls/source_controls.png)

Ошибки чисел/неполнота контрольных входов пособия разобраны в docs/vinogradov/source_*.md; они не спрятаны расширением допуска.

## Предел вывода

Существующее радиальное замыкание проверено независимо. Профилирование не может доказать допустимость незамкнутого потока. Отсутствие решения в конечном поиске не доказывает глобальную математическую невозможность. Типичная LM-кривая IN738C не является аттестованным допуском, а одноосные частоты — полной вибрационной прочностью.
