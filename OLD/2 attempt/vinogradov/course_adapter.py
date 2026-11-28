"""Bind Vinogradov geometry to the original two-stage resource notebook physics.

Gas functions are executed from source cells, never reconstructed from outputs.
All pressures are Pa, temperatures K, lengths m and mass flows kg/s; fuel tables
alone display kg/h. Circumferential outlet velocity is negated when converting
Mikhaltsev's counter-rotation-positive c2u to the signed mechanical frame.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from scipy.optimize import brentq

from .models import FlowStation, RadialSectionInput
from .validation import Check
from .radial_equilibrium_core import pressure_residual


SOURCE_CELLS = (3, 5, 7, 9, 11, 12, 14, 16, 18, 20, 22, 24, 26,
                27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 42)


def air_viscosity_proxy(T):
    """NASA Wind-US SUTHERLAND imperial constants converted to SI, air proxy.

    https://www.grc.nasa.gov/www/winddocs/user/keywords/viscosity.html
    Not validated mixture transport data for combustion products.
    """
    if T <= 100:
        raise ValueError('Sutherland proxy outside documented T>180R domain')
    return 2.329e-8*(1.8*T)**1.5/(1.8*T+216)*(14.59390294/.3048)


def load_course_notebook(path):
    """Execute original definitions/configuration and adopted coupled solve.

    The explicit source-cell boundary excludes plots, historical optimization
    records, sensitivity sweeps and the appended integration (no recursion).
    Source assertions are retained as reportable failures if a real P0 exists.
    Other execution failures propagate with their cell filename/line number.
    """
    path = Path(path).resolve()
    notebook = json.loads(path.read_text(encoding='utf-8'))
    module = ModuleType('_vinogradov_course_' + path.stem)
    module.__file__ = str(path)
    sys.modules[module.__name__] = module  # dataclasses need a genuine module
    namespace = module.__dict__
    namespace['VINOGRADOV_SOURCE_NOTEBOOK'] = path.name
    namespace['VINOGRADOV_SOURCE_ERRORS'] = []
    for index in SOURCE_CELLS:
        cell = notebook['cells'][index]
        if cell['cell_type'] != 'code':
            raise ValueError(f'{path.name}: expected original code cell {index}')
        source = ''.join(cell['source'])
        try:
            exec(compile(source, f'{path.name}:cell{index}', 'exec'), namespace)
        except AssertionError as exc:
            if index != 42 or 'P0' not in namespace:
                raise
            namespace['VINOGRADOV_SOURCE_ERRORS'].append(dict(
                cell=index, name='source assertion', message=str(exc)))
    return namespace


class CourseRadialSampler:
    """Sample the actual source radial solver and source meridional channel."""

    def __init__(self, namespace, *, viscosity=air_viscosity_proxy):
        self.ns = namespace
        self.result = namespace['P0']
        self.cfg = self.result['cfg']
        self.stages = self.result['stages']
        if len(self.stages) != 2:
            raise ValueError('Only the two-stage course turbine is supported')
        self.requirements = namespace['STRENGTH_REQUIREMENTS']
        self.resource_hours = float(self.requirements['life_hours'])
        self.source_notebook = Path(namespace.get('VINOGRADOV_SOURCE_NOTEBOOK',
            f'TVD_1CBK_2st_{self.resource_hours:g}h.ipynb')).name
        self.speed_rpm = float(self.result['geometry'][0])
        if self.speed_rpm != float(namespace['FIXED_SPEED_RPM']):
            raise ValueError('Source fixed resource RPM was changed')
        self.viscosity = viscosity
        self.radials = namespace.get('RADIAL_RESULTS')
        if self.radials is None:
            self.radials = [namespace['radial_stage'](s, self.result['geometry'],
                self.cfg, namespace.get('RADIAL_LAW', 'alpha_const'), points=101)
                for s in self.stages]
        self.design = namespace['PROFILE_ROW_DESIGN']
        self.channel = namespace.get('CHANNEL')
        if self.channel is None:
            self.channel = namespace['meridional_channel'](self.result, self.radials,
                row_widths={name:float(row['bax_m']) for name,row in self.design.items()})
        self.inputs = namespace['cascade_inputs'](self.result, self.radials, self.channel)
        self.blades = self.channel['blade_rows'].set_index('name')
        self._strength_temperature_bounds = {}

    @staticmethod
    def name(index, row):
        return f'{"СА" if row == "SA" else "РК"}{index+1}'

    def total_state(self, index, station, frame_speed=0.):
        alpha = self.stages[index]['alpha']
        ns, cfg = self.ns, self.cfg
        temperature = ns['inv_h'](ns['h'](station.temperature_K, alpha, cfg)
            + station.speed(frame_speed)**2/2, alpha, cfg)
        pressure = ns['pressure_isentropic'](station.temperature_K, temperature,
            station.pressure_Pa, alpha, cfg)
        return float(temperature), float(pressure)

    def _station(self, index, ca, cu, T, p):
        alpha = self.stages[index]['alpha']
        cp = self.ns['cp_true'](float(T), alpha)
        totalT = self.ns['inv_h'](self.ns['h'](T, alpha, self.cfg)
            + (ca*ca+cu*cu)/2, alpha, self.cfg)
        totalP = self.ns['pressure_isentropic'](T,totalT,p,alpha,self.cfg)
        return FlowStation(float(ca),float(cu),float(T),float(p),float(p/(self.cfg.R*T)),
            float(totalT),float(totalP),float(self.viscosity(T)),float(self.cfg.R),
            float(cp/(cp-self.cfg.R)))

    def section(self, index, row, fraction, radial_weight_m):
        if not 0 <= fraction <= 1:
            raise ValueError('span fraction outside [0,1]')
        radial = self.radials[index]
        frame = radial['inlet' if row == 'SA' else 'outlet']
        root, tip = float(frame.r.iloc[0]), float(frame.r.iloc[-1])
        radius = root + fraction*(tip-root)
        mass = float(np.interp(radius,frame.r,frame.mass))/self.stages[index]['G']
        return self._section_mass(index,row,mass,radial_weight_m)

    def _section_mass(self,index,row,mass,weight):
        s,rd = self.stages[index],self.radials[index]
        def at(frame,key):
            return float(np.interp(mass,frame.mass.to_numpy()/s['G'],frame[key]))
        a,b = rd['inlet'],rd['outlet']
        r1,r2 = at(a,'r'),at(b,'r')
        between=self._station(index,at(a,'ca'),at(a,'cu'),at(a,'T'),at(a,'p'))
        downstream=self._station(index,at(b,'ca'),-at(b,'cu'),at(b,'T'),at(b,'p'))
        parts=self.inputs[self.name(index,row)]
        fractions=rd['sections']['mass_fraction'].to_numpy()
        angle0=float(np.interp(mass,fractions,[p['beta_in'] for p in parts]))
        if row=='SA':
            blade=self.blades.loc[self.name(index,row)]
            inlet_radius=math.sqrt((blade.rtip-blade.l_in)**2
                + mass*(blade.rtip**2-(blade.rtip-blade.l_in)**2))
            if index==0:
                station=self.channel['stations'].iloc[0]
                upstream=self._station(index,float(station.ca),0.,float(station['T']),float(station.p))
            else:
                previous=self.radials[index-1]['outlet']
                pmass=previous.mass.to_numpy()/self.stages[index-1]['G']
                ratio=self.stages[index-1]['G']/s['G']
                ca=self.cfg.mu*ratio*float(np.interp(mass,pmass,previous.ca))
                cu=-self.cfg.mu*ratio*float(np.interp(mass,pmass,previous.r*previous.cu))/inlet_radius
                T=self.ns['inv_h'](self.ns['h'](s['T0'],s['alpha'],self.cfg)-(ca*ca+cu*cu)/2,s['alpha'],self.cfg)
                p=self.ns['pressure_isentropic'](s['T0'],T,s['p0'],s['alpha'],self.cfg)
                upstream=self._station(index,ca,cu,T,p)
            inlet,outlet,radius,frame=upstream,between,r1,a
        else:
            inlet,outlet,radius,frame=between,downstream,r2,b
            inlet_radius=r1
        root,tip=float(frame.r.iloc[0]),float(frame.r.iloc[-1])
        # Reaction is evaluated by the original variable-property gas functions.
        Tad=self.ns['isentropic_T'](s['T0'],s['p0'],between.pressure_Pa,s['alpha'],self.cfg)
        Tout=self.ns['isentropic_T'](s['T0'],s['p0'],s['p2'],s['alpha'],self.cfg)
        reaction=self.ns['integral'](Tout,Tad,s['alpha'])/s['H0']
        speed=outlet.speed(radius*self.speed_rpm*2*math.pi/60 if row=='RK' else 0.)
        return RadialSectionInput(index+1,row,radius,root,(root+tip)/2,tip,self.speed_rpm,
            float(reaction),float(s['G']),inlet,outlet,angle0,at(a,'alpha'),at(b,'alpha'),
            at(a,'beta'),at(b,'beta'),outlet.density_kg_m3*speed,float(weight),inlet_radius)

    def row(self,index,row,count=5):
        rd=self.radials[index]
        frame=rd['inlet' if row=='SA' else 'outlet']
        mass=np.linspace(0,1,count)
        radii=np.interp(mass,frame.mass.to_numpy()/self.stages[index]['G'],frame.r)
        weights=np.r_[np.diff(radii)[0]/2,(radii[2:]-radii[:-2])/2,np.diff(radii)[-1]/2]
        return [self._section_mass(index,row,float(f),float(w)) for f,w in zip(mass,weights)]

    def audit(self,index,row):
        rd=self.radials[index]
        # Preserve the explicit numerical acceptance in source notebook §23,
        # cell74; these are solver accuracies, not new design restrictions.
        checks=[Check('source radial mass inlet',abs(float(rd['mass_in_residual'])),1e-8,
                      abs(rd['mass_in_residual'])<=1e-8,'numerical','course §23 cell74',units='kg/s'),
                Check('source radial mass outlet',abs(float(rd['mass_out_residual'])),1e-8,
                      abs(rd['mass_out_residual'])<=1e-8,'numerical','course §23 cell74',units='kg/s'),
                Check('source streamline Euler residual',float(rd['euler_max_residual']),.002,
                      rd['euler_max_residual']<=.002,'numerical','course §23 cell74',units='J/kg')]
        # Independent pressure-equilibrium audit on the actual source mesh.
        frame=rd['inlet' if row=='SA' else 'outlet']
        r=frame.r.to_numpy();p=frame.p.to_numpy();T=frame['T'].to_numpy()
        rhs=frame.cu.to_numpy()**2/r
        residual=pressure_residual(r,p,T,frame.cu.to_numpy(),gas_constant=self.cfg.R)
        scale=max(float(np.max(np.abs(rhs))),1.)
        error=float(np.max(np.abs(residual[1:-1]))/scale)
        checks.append(Check('radial pressure equilibrium relative residual',error,.005,error<=.005,
            'numerical','course §23 cell74 tolerance; accepted pressure_residual on actual grid'))
        stage=self.stages[index];alpha=stage['alpha']
        enthalpy=np.array([self.ns['h'](float(t),alpha,self.cfg) for t in T])
        total=enthalpy+frame.c.to_numpy()**2/2
        expected=self.ns['h'](stage['T0'],alpha,self.cfg)-(stage['Lu'] if row=='RK' else 0.)
        energy_error=float(np.max(np.abs(total-expected)))
        checks.append(Check('variable-cp radial total enthalpy',energy_error,1e-5,
            energy_error<=1e-5,'numerical','original h(T,alpha,cfg); source Lu',units='J/kg'))
        entropy=np.array([self.ns['ds'](float(T[0]),float(t),alpha) for t in T])-self.cfg.R*np.log(p/p[0])
        entropy_error=float(np.max(np.abs(entropy)))
        checks.append(Check('variable-cp radial entropy',entropy_error,1e-5,entropy_error<=1e-5,
            'numerical','original ds(Tref,T,alpha)-R ln(p/pref)',units='J/(kg K)'))
        return dict(checks=checks,pressure_relative_residual=error)

    def throat_state(self,index,station,frame_speed,required_flux):
        """Critical and required throat states using live h(T), s(T), a(T)."""
        ns,cfg=self.ns,self.cfg
        alpha=self.stages[index]['alpha']
        T0,p0=self.total_state(index,station,frame_speed)
        h0=ns['h'](T0,alpha,cfg)
        Tc=brentq(lambda T:h0-ns['h'](T,alpha,cfg)-ns['sound'](T,alpha,cfg)**2/2,180.,T0)
        def flux(T):
            p=ns['pressure_isentropic'](T0,T,p0,alpha,cfg)
            v=math.sqrt(max(0.,2*(h0-ns['h'](T,alpha,cfg))))
            return p/(cfg.R*T)*v
        critical=flux(Tc)
        possible=required_flux<=critical
        temperature=brentq(lambda T:flux(T)-required_flux,Tc,T0) if possible else None
        mach=(math.sqrt(2*(h0-ns['h'](temperature,alpha,cfg)))/ns['sound'](temperature,alpha,cfg)
              if possible else None)
        return dict(possible=bool(possible),required_flux_kg_m2_s=float(required_flux),
            critical_flux_kg_m2_s=float(critical),required_mach=mach,
            stagnation_temperature_K=T0,stagnation_pressure_Pa=p0,
            critical_temperature_K=float(Tc),source='original variable-cp h, pressure_isentropic, sound')

    def strength(self, profiles, *, rotors=(1, 2)):
        """Assess only requested rotors with native UKR, gas bending and LM.

        Five actual polygons supply trapezoidal centrifugal loads; native UKR
        uses the actual root/mid/tip contours for quadratic loads and bending.
        Missing requested geometry leaves the total margin unknown. An absent
        unrequested sibling is irrelevant. Gas-only temperature bounds are
        reused within this sampler, never across new gas-state samplers.
        """
        from .blade_strength import centrifugal_loads
        rotors = tuple(rotors)
        if not rotors or any(rotor not in (1, 2) for rotor in rotors) or len(set(rotors)) != len(rotors):
            raise ValueError('Request distinct rotor numbers 1 and/or 2')
        ns = self.ns
        available = [rotor for rotor in rotors
                     if all((f'РК{rotor}', j) in profiles for j in range(5))]
        assessment = dict(table=ns['pd'].DataFrame(), passed=False, min_margin=math.nan,
            rotor_results={}, temperature_bounds={}, reasons=[],
            computed_rotors=[], incomplete_rotors=[rotor for rotor in rotors if rotor not in available])
        assessment['reasons'].extend(f'РК{rotor}: missing five-section geometry'
                                     for rotor in assessment['incomplete_rotors'])
        # Retain the native candidate_strength material/curve preconditions,
        # without invoking its necessarily whole-turbine geometry calculation.
        required = ('material', 'life_hours', 'C_LM', 'curve_points',
                    'required_margin', 'density_kg_m3')
        missing = [key for key in required
                   if key not in self.requirements or self.requirements[key] is None]
        if missing:
            assessment['reasons'].append('Не заданы требования: ' + ', '.join(missing))
            return assessment
        try:
            if not isinstance(self.requirements['material'], str) or not self.requirements['material'].strip():
                raise ValueError('Нужна явно идентифицированная марка материала')
            life, C = float(self.requirements['life_hours']), float(self.requirements['C_LM'])
            density = float(self.requirements['density_kg_m3'])
            margin = float(self.requirements['required_margin'])
            if (not np.isfinite(life) or life <= 0 or not np.isfinite(C) or
                    C + np.log10(life) <= 0):
                raise ValueError('Ресурс должен быть >0 ч, C_LM конечен, C_LM+log10(ресурс)>0')
            if not np.isfinite(density) or density <= 0 or not np.isfinite(margin) or margin <= 0:
                raise ValueError('Плотность и требуемый запас должны быть конечными и >0')
            points = np.asarray(self.requirements['curve_points'], dtype=float)
            if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or
                    not np.all(np.isfinite(points)) or np.any(points[:, 0] <= 0) or
                    np.any(points[:, 1] <= 0) or np.any(np.diff(points[:, 0]) <= 0) or
                    np.any(np.diff(points[:, 1]) >= 0)):
                raise ValueError('LM-кривая: возрастающие P/1000>0 и убывающие σ>0, без экстраполяции')
            if len(self.radials) != 2:
                raise ValueError('Проверка требует обе ступени и радиальные состояния обоих РК')
        except (KeyError, TypeError, ValueError, IndexError, FloatingPointError, OverflowError) as exc:
            assessment['reasons'].append(str(exc))
            return assessment
        for rotor in available:
            try:
                sections=[profiles[(f'РК{rotor}',j)] for j in range(5)]
                radii=np.array([p['radius'] for p in sections])
                areas = []
                for section in sections:
                    polygon = np.asarray(section['polygon'], dtype=float)
                    if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 4 or
                            not np.isfinite(polygon).all()):
                        raise ValueError('Недопустимый реальный контур')
                    areas.append(ns['ukr_section_properties'](polygon)['F'])
                areas = np.asarray(areas)
                if not np.isfinite(radii).all() or np.any(radii <= 0):
                    raise ValueError('Недопустимый радиус')
                loads=centrifugal_loads(radii,areas,self.speed_rpm,density)
                preliminary=ns['pd'].DataFrame([dict(rotor=rotor,section=j,r=r,F=A,N=N,sigma_tension=s)
                    for j,(r,A,N,s) in enumerate(zip(radii,areas,loads['force_N'],loads['stress_Pa']))])
                if rotor not in self._strength_temperature_bounds:
                    self._strength_temperature_bounds[rotor] = ns['candidate_temperature_bound'](
                        self.radials[rotor-1], self.stages[rotor-1], self.cfg)
                bound = self._strength_temperature_bounds[rotor]
                ukr=ns['ukr_rotor_strength'](self.result,profiles,self.radials,self.cfg,
                    self.requirements,rotor=rotor,load_model='radial')
                stresses=[dict(method='trapezoid_5_tension',section=j,section_name=f'Сечение {j}',r_m=r,
                    sigma_tension_MPa=s/1e6,sigma_bending_min_MPa=np.nan,sigma_bending_max_MPa=np.nan,
                    sigma_normal_min_MPa=s/1e6,sigma_normal_max_MPa=max(0.,s)/1e6)
                    for j,(r,s) in enumerate(zip(radii,loads['stress_Pa']))]
                for j,section in zip((0,2,4),ukr['sections']):
                    total=np.asarray(section['sigma_total'],float);bending=np.asarray(section['sigma_bending'],float)
                    if not np.isfinite(total).all() or not np.isfinite(bending).all():
                        raise ValueError('Nonfinite native UKR normal stress')
                    stresses.append(dict(method='quadratic_3_tension_gas_bending',section=j,
                        section_name=section['name'],r_m=float(section['r']),
                        sigma_tension_MPa=section['sigma_tension']/1e6,
                        sigma_bending_min_MPa=float(bending.min())/1e6,sigma_bending_max_MPa=float(bending.max())/1e6,
                        sigma_normal_min_MPa=float(total.min())/1e6,sigma_normal_max_MPa=max(0.,float(total.max()))/1e6))
                rows=ns['_candidate_lm_rows'](rotor,stresses,bound,self.requirements,points)
                assessment['table']=ns['pd'].concat([assessment['table'],ns['pd'].DataFrame(rows)],ignore_index=True)
                ukr.update(trapezoid_table=preliminary,lm_table=ns['pd'].DataFrame(rows))
                assessment['rotor_results'][rotor]=ukr
                assessment['temperature_bounds'][rotor]=bound
                assessment['reasons'].extend(
                    f"РК{rotor}, {row['method']}, {row['section_name']}: {row['reason']}"
                    for row in rows if not row['passed'])
            except (KeyError,TypeError,ValueError,IndexError,FloatingPointError,OverflowError) as exc:
                assessment['reasons'].append(f'РК{rotor}: {exc}')
        assessment['computed_rotors']=sorted(assessment['rotor_results'])
        table = assessment['table']
        complete = len(assessment['computed_rotors']) == len(rotors)
        if complete and not table.empty:
            margins = table['margin'].to_numpy(dtype=float)
            assessment['min_margin'] = float(np.min(margins)) if not np.isnan(margins).any() else math.nan
            assessment['passed'] = bool(not assessment['reasons'] and table['passed'].all())
        return assessment
