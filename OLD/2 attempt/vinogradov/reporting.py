"""Actual calculated surfaces only; failed rows are explained, never redrawn.

Vinogradov profile/channel diagrams plus signed velocity triangles, real
section thickness/curvature, span properties/strength, meridional and 3D view.
"""
from pathlib import Path
import math
import numpy as np
from .curves import curvature
from .blade_geometry import medial_line
from .velocity_triangles import velocity_triangles


def render_report(report,row_objects,out,namespace):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    out=Path(out);images=[]
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/name,dpi=160);plt.close(fig);images.append(name)
    for row in row_objects:
        evaluations=row['evaluations'];indices=[0,len(evaluations)//2,len(evaluations)-1]
        fig,axs=plt.subplots(2,3,figsize=(14,9))
        state=next(record for record in report['rows'] if record['id']==row['id'])
        fig.suptitle(f'{row["id"]}: {"PRELIMINARY" if state["passed"] else "FAILED — diagnostic geometry, not an accepted blade"}',
                     color='black' if state['passed'] else 'darkred',fontsize=12)
        for col,j in enumerate(indices):
            v=evaluations[j];p=v['profile'];sp=v['spec'];s=v['section'];channel=v['channel']
            poly=p.contour()*1000;ax=axs[0,col]
            ax.plot(poly[:,1],-poly[:,0]);ax.fill(poly[:,1],-poly[:,0],alpha=.12)
            ax.set_title(f'{row["id"]} / {("Root","Mid","Tip")[col]}\nr={s.radius_m*1000:.2f} mm, Z={row["count"]}')
            ax.set(xlabel='Circumferential, mm',ylabel='-Axial, mm');ax.axis('equal');ax.grid(alpha=.2)
            ax=axs[1,col]
            for shift in (0,sp.pitch_m):
                points=p.contour()+[0,shift];ax.plot(points[:,0]*1000,points[:,1]*1000,lw=1)
            for circle in channel.circles[::max(1,len(channel.circles)//12)]:
                ax.add_patch(Circle(circle.center_m*1000,circle.radius_m*1000,fill=False,lw=.5,color='green'))
            ax.plot(channel.centers_m[:,0]*1000,channel.centers_m[:,1]*1000,'k--',lw=.8)
            ax.plot([channel.throat.suction_contact_m[0]*1000,channel.throat.pressure_contact_m[0]*1000],
                    [channel.throat.suction_contact_m[1]*1000,channel.throat.pressure_contact_m[1]*1000],'r-',lw=2)
            ax.set(xlabel='Axial, mm',ylabel='Circumferential, mm',title=f'Inscribed channel; a={channel.throat_width_m*1000:.3f} mm')
            ax.axis('equal');ax.grid(alpha=.2)
        save(fig,f'{row["id"]}_profiles_channels.png')
        fig,axs=plt.subplots(3,3,figsize=(14,11))
        for col,j in enumerate(indices):
            v=evaluations[j];p=v['profile'];s=v['section'];ch=v['channel']
            medial=medial_line(p,81);centers=medial['centers_m']
            axs[0,col].plot(centers[:,0]*1000,2*medial['radii_m']*1000)
            axs[0,col].set(xlabel='Axial, mm',ylabel='Normal thickness, mm')
            t=np.linspace(.001,.999,401)
            for side,name in [(p.suction,'suction'),(p.pressure,'pressure')]:
                # Separate analytic pieces: do not hide curvature jumps by joining
                # unrelated parameterization derivatives across a composite joint.
                for k,segment in enumerate(side.segments):
                    arc=segment.evaluate(t)
                    axs[1,col].plot(arc[:,0]*1000,curvature(segment,t),label=name if k==0 else None)
            axs[1,col].set(xlabel='Axial, mm',ylabel='Signed curvature, 1/m');axs[1,col].legend(fontsize=8)
            axs[2,col].plot(ch.arc_length_m*1000,ch.width_m*1000)
            for a,b in ch.diffuser_intervals_m:axs[2,col].axvspan(a*1000,b*1000,color='red',alpha=.15)
            axs[2,col].set(xlabel='Channel arc length, mm',ylabel='Channel width, mm')
            for ax in axs[:,col]:ax.grid(alpha=.2)
        save(fig,f'{row["id"]}_thickness_curvature_width.png')
        blade=row['blade'];fig,axs=plt.subplots(1,3,figsize=(14,4))
        span=blade.radii_m*1000
        axs[0].plot(span,blade.areas_m2*1e6,'o-',label='actual sections')
        record=next(r for r in report['rows'] if r['id']==row['id'])
        for law,areas in record['area_laws'].items():
            axs[0].plot(np.linspace(span[0],span[-1],len(areas)),np.asarray(areas)*1e6,'--',label=law)
        axs[0].set(xlabel='Radius, mm',ylabel='Area, mm2');axs[0].legend(fontsize=7)
        axs[1].plot(blade.centroids_m[:,0]*1000,blade.centroids_m[:,1]*1000,'o-')
        axs[1].set(xlabel='Centroid axial, mm',ylabel='Centroid circumferential, mm')
        if 'strength' in record:
            rows=record['strength']['rows']
            margin=record['strength']['required_lm_margin']
            for method in dict.fromkeys(v['method'] for v in rows):
                group=sorted((v for v in rows if v['method']==method),key=lambda v:v['radius_m'])
                r=np.array([v['radius_m'] for v in group])*1000
                axs[2].plot(r,[v['normal_max_Pa']/1e6 for v in group],'o-',label=method+' max')
                if all('normal_min_Pa' in v for v in group):
                    axs[2].plot(r,[v['normal_min_Pa']/1e6 for v in group],'o:',label=method+' min')
                axs[2].plot(r,[v['lm_allowable_Pa']/margin/1e6 for v in group],'--',label=f'LM/{margin:g}')
            axs[2].set_title('Stationary SA gas bending' if row['id'].startswith('SA')
                            else 'Source 5-trapezoid and 3-quadratic normal stress')
            axs[2].set(xlabel='Radius, mm',ylabel='Normal stress, MPa')
            if rows:axs[2].legend(fontsize=7)
        else:
            for j in indices:
                v=evaluations[j];s=v['section']
                cu1=s.inlet.circumferential_m_s;ca1=s.inlet.axial_m_s
                cu2=s.outlet.circumferential_m_s;ca2=s.outlet.axial_m_s
                axs[2].plot([0,cu1,cu2],[0,ca1,ca2],label=f'r={s.radius_m*1000:.1f}')
            axs[2].set(xlabel='Signed c_theta, m/s',ylabel='c_axial, m/s');axs[2].legend(fontsize=7)
        for ax in axs:ax.grid(alpha=.2)
        save(fig,f'{row["id"]}_span_properties.png')
    fig,ax=plt.subplots(figsize=(12,5))
    stages=namespace['P0']['stages']
    for i,stage in enumerate(stages):
        for row,D,h in [('SA',stage['Dc'],stage['l_c']),('RK',stage['Dl'],stage['l_l'])]:
            record=next(v for v in report['rows'] if v['stage']==i+1 and v['row']==row)
            x=record['axial_start_m'];width=record['axial_envelope_m']
            ax.axvspan(x*1000,(x+width)*1000,alpha=.1,color='blue' if row=='SA' else 'orange')
            ax.annotate(f'{row}{i+1}',((x+width/2)*1000,(D+h)*500),xytext=(0,12),textcoords='offset points',ha='center')
    contour=report['meridional_channel']['contour']
    ax.plot(np.asarray(contour['x'])*1000,np.asarray(contour['rhub'])*1000,'k-',lw=1)
    ax.plot(np.asarray(contour['x'])*1000,np.asarray(contour['rtip'])*1000,'k-',lw=1)
    ax.set(xlabel='Axial, mm',ylabel='Radius, mm',title=report['meridional_channel']['basis'])
    ax.margins(y=.25);save(fig,'meridional.png')
    fig,axs=plt.subplots(1,len(stages),figsize=(5*len(stages),4),squeeze=False)
    for i,s in enumerate(stages):
        tr=velocity_triangles(s['c1'],s['a1'],s['u1'],s['w2'],s['beta2'])
        ax=axs[0,i]
        for data,label in [(tr.inlet_plot_m_s,'inlet'),(tr.outlet_plot_m_s,'outlet')]:ax.plot(data[:,0],data[:,1],'o-',label=label)
        ax.set(title=f'Stage{i+1} velocity triangles',xlabel='Book -circumferential, m/s',ylabel='Book -axial, m/s');ax.axis('equal');ax.legend();ax.grid(alpha=.2)
    save(fig,'velocity_triangles.png')
    if row_objects:
        fig=plt.figure(figsize=(12,8));ax=fig.add_subplot(111,projection='3d')
        for row in row_objects:
            # Complete rows; decimate contour points only for visual rendering.
            for blade in row['blade'].ring(row['count']):
                for p in (blade[0],blade[len(blade)//2],blade[-1]):ax.plot(p[::4,0]*1000,p[::4,1]*1000,p[::4,2]*1000,lw=.25)
        ax.set(xlabel='Axial, mm',ylabel='Y, mm',zlabel='Z, mm',title='Calculated blade rows; EXPERIMENTAL geometry')
        save(fig,'turbine_3d.png')
    text=['# Vinogradov — actual TWO-stage course resource calculation','',
          f'Source: **{report["source_notebook"]}**; resource: **{report["resource_hours"]:g} h**; stages: **2**.',
          f'Fixed source RPM: **{report["speed_rpm"]:g}**.',
          f'Engineering admissible: **{report["passed"]}**.',
          f'Constructed rows: {len(row_objects)}/4.',
          '', 'A failed finite search is not a proof of global infeasibility. No design limit was relaxed.',
          '', '## Quantified violations','']
    for row in report['rows']:
        text.append(f'### {row["id"]}')
        for violation in row['violations']:text.append('- '+str(json_safe(violation)))
        if not row['violations']:text.append('- No recorded row violation; global checks still required.')
    text+=['','## Global/source violations','']
    text+=['- '+str(json_safe(v)) for v in report['violations']]
    text+=['','## Empirical guidance — not added hard limits','']
    text+=['- '+str(json_safe(v)) for v in report.get('warnings',[])]
    text+=['','## Figures','']+[f'![{name}]({name})' for name in images]
    text+=['','## Scope and missing physical evidence','']+['- '+v for v in report['limitations']]
    (out/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    report['figures']=images


def json_safe(data):
    from .cad_export import json_value
    return json_value(data)


def draw_course_profiles(namespace, output_dir):
    """Only textbook contours at root/mid/tip; no gas or strength optimization."""
    from dataclasses import replace
    from .course_adapter import CourseRadialSampler
    from .integration import _source_specs, _source_profiles
    from .cascade_geometry import build_profile
    from .validation import validate_profile
    from .models import ConstructionError
    from .cad_export import export_json
    import matplotlib.pyplot as plt

    sampler=CourseRadialSampler(namespace)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    figure,axes=plt.subplots(4,3,figsize=(13,15),layout='none')
    rows=[];indices=(0,2,4);labels=('Корень','Среднее','Периферия')
    try:
        for i in range(2):
            for row in ('SA','RK'):
                name=sampler.name(i,row);sections=sampler.row(i,row,5)
                seeds=_source_specs(sampler,i,row,sections,_source_profiles(sampler),{})
                seeds=[seeds[j] for j in indices]
                # Gamma is an independent geometric input, not the old cubic
                # profile's stagger. The tangent bisector starts the RK shape.
                if row=='RK':
                    seeds=[replace(s,stagger_deg=90+(s.outlet_metal_deg-s.inlet_metal_deg)/2)
                           for s in seeds]
                chosen=None;best_jump=math.inf
                for offset in (0.,5.,-5.,10.,-10.,15.,-15.):
                    specs=[replace(s,stagger_deg=s.stagger_deg+offset) for s in seeds]
                    try:
                        profiles=[build_profile(s,'double_parabolic') for s in specs]
                        checks=[validate_profile(p,s.pitch_m,samples=301)
                                for p,s in zip(profiles,specs)]
                        chords=[float(np.linalg.norm(p.trailing_edge.center-p.leading_edge.center))
                                for p in profiles]
                        direction=(chords[0]>chords[-1] if row=='RK' else chords[0]<chords[-1])
                        if all(v['passed'] for v in checks) and direction:
                            # Keep the exact source curves, not a cosmetic
                            # spline. Prefer a milder internal curvature jump
                            # so a formally G1 joint does not form a sharp knee.
                            jump=max(float(abs(curvature(p.suction.segments[0],1)-
                                curvature(p.suction.segments[1],0))*chord)
                                for p,chord in zip(profiles,chords))
                            if jump<best_jump:
                                chosen=specs,profiles,chords;best_jump=jump
                            if row=='SA':break
                    except (ConstructionError,ValueError,np.linalg.LinAlgError):
                        continue
                if chosen is None:
                    raise ConstructionError(f'{name}: no valid drawing with the supplied dimensions and angles')
                specs,profiles,chords=chosen
                panel=2*i+(row=='RK');points=[]
                for col,(j,label,sp,p,chord) in enumerate(zip(indices,labels,specs,profiles,chords)):
                    contour=(p.contour(401)-p.leading_edge.center)*1000
                    xy=np.column_stack((contour[:,1]*(-1 if row=='SA' else 1),-contour[:,0]))
                    points.append(xy);ax=axes[panel,col]
                    ax.plot(xy[:,0],xy[:,1],color='navy',lw=1.4)
                    ax.fill(xy[:,0],xy[:,1],color='steelblue',alpha=.15)
                    ax.set_title(f'{name} — {label}\nr={sections[j].radius_m*1000:.2f} мм; хорда={chord*1000:.2f} мм')
                    ax.set(xlabel='Окружная координата, мм',ylabel='−Осевая координата, мм')
                    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
                    rows.append(dict(row=name,section=label,radius_m=sections[j].radius_m,
                        chord_m=chord,count=sampler.design[name]['Z'],specification=sp))
                # Identical scales: an auto-zoomed tip must not look larger
                # than the root merely because its subplot has tighter limits.
                all_points=np.vstack(points);lo=all_points.min(axis=0);hi=all_points.max(axis=0)
                margin=max(float(np.max(hi-lo))*.08,.5)
                for ax in axes[panel]:
                    ax.set_xlim(lo[0]-margin,hi[0]+margin);ax.set_ylim(lo[1]-margin,hi[1]+margin)
        resource=int(sampler.requirements['life_hours'])
        figure.suptitle(f'Профили по Виноградову — {resource} ч; n={sampler.speed_rpm:g} об/мин',fontsize=15,y=.975)
        figure.tight_layout(rect=(0,.01,1,.94),h_pad=2.5)
        path=out/'profiles.png';figure.savefig(path,dpi=160)
        export_json(out/'profiles.json',dict(resource_hours=resource,rows=rows,
            source='Vinogradov tables 4.1/5.1, pp.61–76',
            scope='Profile drawings only. Existing gas states and blade counts unchanged; no new strength or loss qualification.',
            stagger_choice='RK tangent bisector plus a common small stagger offset selected for valid geometry and lower internal curvature jump. SA starts with existing geometric angle.'))
        return path
    finally:
        plt.close(figure)
