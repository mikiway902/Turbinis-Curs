#!/usr/bin/env python3
"""Run actual TWO-stage resource notebook physics and Vinogradov profiling.

Example: python run_vinogradov.py --resource 10000 --iterations 3
The 110 kW target is AFTER the gearbox; resource RPM remains source-fixed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from vinogradov.cad_export import export_json
from vinogradov.course_adapter import load_course_notebook
from vinogradov.integration import design_from_namespace
from vinogradov.reference_cases import control_report


ROOT=Path(__file__).resolve().parent
RESOURCES=(300,2000,10000)


def run_design(output_dir=None,*,resource_hours=10000,iterations=3,
               geometry_iterations=4,count_half_window=1,
               optimize_geometry=True,export_cad=True):
    """Recompute a real source design; feed back each stage's own phi/psi.

    A failed/incomplete geometry has no fabricated coefficient. It stops
    feedback, preserving the last genuine gas state and quantified row errors.
    """
    if resource_hours not in RESOURCES:
        raise ValueError(f'resource_hours must be one of {RESOURCES}')
    if iterations<1 or geometry_iterations<1 or count_half_window<0:
        raise ValueError('positive iteration counts and nonnegative count window required')
    namespace=load_course_notebook(ROOT/f'TVD_1CBK_2st_{resource_hours}h.ipynb')
    return run_design_from_namespace(namespace,output_dir,iterations=iterations,
        geometry_iterations=geometry_iterations,count_half_window=count_half_window,
        optimize_geometry=optimize_geometry,export_cad=export_cad)


def run_design_from_namespace(namespace,output_dir=None,*,iterations=3,
                              geometry_iterations=4,count_half_window=1,
                              optimize_geometry=True,export_cad=True,initial_specifications=None):
    """Couple an actual already-computed course P0; never reload its notebook.

    Retain every iteration's geometry, rejection archive and drawings.
    Carry geometry as initializers, not as an exemption from validation.
    Update source axial row widths/counts before the next gas/channel review.
    """
    resource_hours=namespace['STRENGTH_REQUIREMENTS']['life_hours']
    if resource_hours not in RESOURCES:
        raise ValueError(f'resource_hours must be one of {RESOURCES}')
    resource_hours=int(resource_hours)
    if iterations<1 or geometry_iterations<1 or count_half_window<0:
        raise ValueError('positive iteration counts and nonnegative count window required')
    out=Path(output_dir) if output_dir is not None else ROOT/'exports'/'vinogradov'/f'{resource_hours}h'
    out.mkdir(parents=True,exist_ok=True)
    controls=control_report(ROOT/'exports'/'vinogradov'/'controls')
    source=Path(namespace.get('VINOGRADOV_SOURCE_NOTEBOOK',ROOT/f'TVD_1CBK_2st_{resource_hours}h.ipynb'))
    warm={} if initial_specifications is None else initial_specifications
    sequence=[];converged=False;reason='coupled coefficient tolerance not reached'
    tolerance=float(namespace['LOSS_MODEL']['coefficient_tolerance'])
    relaxation=.55  # same explicit relaxation as original coupled_turbine_design
    report=None
    for iteration in range(iterations):
        iteration_out=out/'iterations'/f'{iteration:03d}'
        report=design_from_namespace(namespace,dict(output_dir=str(iteration_out),
            optimize_geometry=optimize_geometry,max_iterations=geometry_iterations,
            count_half_window=count_half_window,radial_sections=5,channel_samples=41,
            adaptive_sections=True,max_radial_sections=65,cad_tolerance_m=2e-5,
            export_step=export_cad,initial_specifications=warm))
        cfg=namespace['P0']['cfg']
        old=np.array([*cfg.phi,*cfg.psi],dtype=float)
        coefficients=report['velocity_coefficients']
        new=np.array([*coefficients['phi'],*coefficients['psi']],dtype=float)
        complete=bool(new.shape==(4,) and np.isfinite(new).all() and
                      np.all((new>0)&(new<=1)))
        error=float(np.max(np.abs(new-old))) if complete else None
        sequence.append(dict(iteration=iteration,source_notebook=source.name,
            resource_hours=resource_hours,stage_count=2,
            power_after_gearbox_W=float(namespace['P0']['power_W']),
            speed_rpm=float(namespace['P0']['geometry'][0]),
            air_mass_flow_kg_s=float(namespace['P0']['G']),
            gas_mass_flows_kg_s=[float(s['G']) for s in namespace['P0']['stages']],
            prescribed={'phi':list(cfg.phi),'psi':list(cfg.psi)},
            measured=coefficients,coefficient_error=error,passed=report['passed'],
            report_path=str(iteration_out/'report.json')))
        if not complete:
            reason='feedback unavailable: one or more actual row loss evaluations failed'
            break
        if error<=tolerance:
            converged=True;reason='per-stage source coefficient tolerance reached'
            break
        if iteration+1==iterations:
            break
        warm={row['id']:row['specifications'] for row in report['rows']}
        for row in report['rows']:
            name=('СА' if row['row']=='SA' else 'РК')+str(row['stage'])
            registry=namespace['PROFILE_ROW_DESIGN'][name]
            registry['Z']=row['count']
            registry['bax_m']=max(spec['axial_width_m'] for spec in warm[row['id']])
            for shape,spec in zip(registry['sections'],warm[row['id']]):
                shape['bax_m']=spec['axial_width_m']
        values=old+relaxation*(new-old)
        current=replace(cfg,phi=tuple(values[:2]),psi=tuple(values[2:]))
        # Source evaluate solves flow/power again with its own variable-cp gas
        # functions. There is no optimizer import, RPM search or averaged row.
        try:
            result=namespace['evaluate'](namespace['P0']['geometry'],current,
                exit_area=namespace.get('P0_EXIT_AREA_M2')
                    if namespace.get('P0_MODE')=='FIXED_AREA' else None)
        except (ValueError,ArithmeticError,RuntimeError) as exc:
            reason='source feedback gas evaluation failed: '+str(exc)
            report['violations'].append(dict(name='source coupled gas failure',
                message=str(exc),attempted_phi=list(current.phi),attempted_psi=list(current.psi)))
            break
        if float(result['geometry'][0])!=float(namespace['FIXED_SPEED_RPM']):
            raise ValueError('feedback changed fixed resource RPM')
        namespace.update(P0=result,C=current,GEOMETRY=result['geometry'])
        namespace.pop('RADIAL_RESULTS',None);namespace.pop('CHANNEL',None)
    if not converged:
        report['violations'].append(dict(name='coupled source coefficient consistency',
            reason=reason,value=sequence[-1]['coefficient_error'],limit=tolerance))
    report.update(converged=converged,passed=bool(report['passed'] and converged),
        status='FAILED' if not (report['passed'] and converged) else 'PRELIMINARY_PASS',
        reason=reason,gas_iterations=sequence,source_controls=controls,
        final_blading_report=str(iteration_out/'report.json'))
    export_json(out/'report.json',report)
    export_json(out/'coupled_report.json',{key:report[key] for key in
        ('source_notebook','resource_hours','stage_count','speed_rpm','converged','passed',
         'status','reason','gas_iterations','source_controls','final_blading_report')})
    with (out/'REPORT.md').open('w',encoding='utf-8') as stream:
        stream.write(f'\n## Coupled source feedback\n\nConverged: **{converged}**. '
            f'Final admissible: **{report["passed"]}**.\n\n{reason}.\n'
            '\n110 kW is after the gearbox; CBK is only the cycle compressor. '
            'Resource RPM, material, life and LM reserve are inherited from the source.\n'
            f'\nFinal detailed iteration: `{iteration_out/"REPORT.md"}`. '
            'Earlier rejected variants remain in the numbered iteration directories.\n')
    print(f'Source: {source.name}; stages=2; resource={resource_hours} h; '
          f'110 kW AFTER gearbox; fixed RPM={report["speed_rpm"]:g}\n'
          f'Report: {out/"report.json"}\nConverged={converged}; admissible={report["passed"]}',flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resource',type=int,choices=RESOURCES,default=10000)
    parser.add_argument('--iterations',type=int,default=3)
    parser.add_argument('--geometry-iterations',type=int,default=4)
    parser.add_argument('--count-half-window',type=int,default=1)
    parser.add_argument('--source-only',action='store_true')
    parser.add_argument('--no-geometry-search',action='store_true')
    parser.add_argument('--no-step',action='store_true')
    args=parser.parse_args()
    if args.source_only:
        result=control_report((args.output/'controls') if args.output else ROOT/'exports'/'vinogradov'/'controls')
        print(f'Source comparisons={len(result["comparisons"])}; '
              f'documented source discrepancies={len(result["source_comparison_failures"])}')
    else:
        run_design(args.output,resource_hours=args.resource,iterations=args.iterations,
            geometry_iterations=args.geometry_iterations,count_half_window=args.count_half_window,
            optimize_geometry=not args.no_geometry_search,export_cad=not args.no_step)


if __name__=='__main__':
    main()
