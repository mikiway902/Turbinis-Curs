"""Literal printed controls with absolute/relative error, not re-pinned outputs.

Every expected value below is transcribed from the inspected source page.
Contradictions remain failed SOURCE comparisons with mathematical explanations;
software assertions test the governing identities separately.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from .models import ProfileSpec
from .parabolic_profile import build_scheme,build_parabolic_profile,maximum_thickness,parabolic_envelope
from .double_parabolic_profile import prescribed_thickness_construction,build_double_parabolic_profile
from .active_profile import active_profile_construction,build_active_profile
from .section_properties import section_properties
from .channel_convergence import analyze_channel
from .cascade_throat import source_throat_width
from .velocity_triangles import velocity_triangles,elementary_stage_layout
from .cad_export import export_csv,export_json,export_dxf,export_step
from .blade_stacking import stack_sections


def control_report(output_dir):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
    def record(table,page,name,actual,printed,units,tolerance,note=''):
        actual=float(actual);error=actual-printed
        rows.append(dict(table=table,page=page,quantity=name,actual=actual,printed=printed,
            absolute_error=error,absolute_error_magnitude=abs(error),relative_error=error/printed if printed else None,
            units=units,tolerance=tolerance,within_printed_tolerance=abs(error)<=tolerance,note=note))
    chord=.1/math.sin(math.radians(36.33))
    scheme_spec=ProfileSpec(.1,36.33,90,14,.595*chord,.0243*chord,.006025*chord)
    scheme=build_scheme(scheme_spec)
    for name,value,expected in [('chord',scheme['chord_m']*1000,168.795),('pressure_radius',scheme['pressure_radius_m']*1000,233.025),
                                ('B.x',scheme['B'][0]*1000,67.623),('B.y',scheme['B'][1]*1000,140.181)]:
        record('2.1',21 if name=='chord' else 28,name,value,expected,'mm',.0006)
    p31_spec=ProfileSpec(.1,36.33,90,17,.685*chord,.0243*chord,.006025*chord)
    p31=build_parabolic_profile(p31_spec)
    record('3.1',43,'P.x',p31.metadata['throat_point_m'][0]*1000,66.932,'mm',.006,'exact quadratic vs sparse spline')
    record('3.1',43,'back_angle',p31.metadata['back_angle_deg'],13.325,'deg',.005,'exact quadratic vs sparse spline')
    record('3.1',43,'throat_circle_radius',p31.metadata['throat_circle_radius_m']*1000,36.825,'mm',.0006,
           'SOURCE INCONSISTENCY: printed Euclidean-distance result contradicts printed points')
    chord32=.1/math.sin(math.radians(47.))
    p32=build_parabolic_profile(ProfileSpec(.1,47,101,25,.63*chord32,.031*chord32,.0095*chord32))
    thickness=maximum_thickness(p32)
    # The routine reports diameter_m; see source-specific common-normal test.
    diameter=thickness['diameter_m']
    record('3.2',46,'maximum_thickness',diameter*1000,16.003,'mm',.006,'source linear zero interpolation vs exact normals')
    c=.1/math.sin(math.radians(40))
    c9017_spec=ProfileSpec(.1,40,90,17,.69*c,.024*c,.006*c)
    c9017=build_parabolic_profile(c9017_spec);prop=section_properties(c9017,count_per_side=2001)
    for name,value,expected,units,tolerance in [('area',prop.area_m2*1e6,2659,'mm2',26.59),
        ('centroid_x',prop.centroid_m[0]*1000,51.457,'mm',.5),('centroid_y',prop.centroid_m[1]*1000,66.011,'mm',.5),
        ('principal_min',prop.principal_min_m4*1e12,184100,'mm4',.03*184100),
        ('principal_max',prop.principal_max_m4*1e12,3323000,'mm4',.02*3323000)]:
        record('3.3',48,name,value,expected,units,tolerance)
    channel=analyze_channel(c9017,c9017_spec.pitch_m,samples=61,wall_samples=1001)
    design=source_throat_width(c9017_spec.pitch_m,17)
    record('3.4',54,'K_prescribed',design/.066,.453,'1',.0005,
           'prescribed throat, NOT measured geometry; measured throat separately in report')
    record('3.4',54,'achieved_vs_prescribed_throat',channel.throat_width_m*1000,design*1000,'mm',.001,
           'SOURCE DESIGN DOES NOT CLOSE: actual inscribed throat differs from prescribed t*sin(beta)/m')
    triangles=velocity_triangles(200,17,100,150,35)
    for name,value,expected in [('c1_axial',triangles.absolute_inlet_m_s[0],58.47434094),
                               ('w2_theta',triangles.relative_outlet_m_s[1],-122.87280664)]:
        record('3.5',57,name,value,expected,'m/s',1e-8)
    # Sequential Mathcad assignments, table3.6: unassigned alpha1=17 and
    # Kt=.69 retain their stator values; this assumption is recorded explicitly.
    rotor_chord=.1/math.sin(math.radians(45))
    rotor_spec=ProfileSpec(.1,45,79,17,.69*rotor_chord,.04*rotor_chord,.008*rotor_chord)
    rotor=build_parabolic_profile(rotor_spec)
    layout=elementary_stage_layout(c9017,rotor,stator_pitch_m=c9017_spec.pitch_m,rotor_pitch_m=rotor_spec.pitch_m,triangles=triangles)
    record('3.6',60,'inlet_triangle_origin_x',layout.inlet_triangle_m[0,0]*1000,100,'mm',1e-10,
           'sequential assignment interpretation: alpha1=17,Kt=.69,width=100,VA=100 retained')
    p45_spec=ProfileSpec(.1,40,90,17,.095677,.0243*c,.006025*c,relative_thickness=.106,thickness_location=.245)
    p45=build_double_parabolic_profile(p45_spec);data=prescribed_thickness_construction(p45_spec)
    for name,value,expected in [('thickness_center_y',data['thickness_center'][1]*1000,88.110),
                              ('suction_contact_x',data['suction_contact'][0]*1000,29.133)]:
        record('4.1',70,name,value,expected,'mm',.0006)
    record('4.1',69,'pressure_contact_x',data['pressure_contact'][0]*1000,20.527,'mm',.0006,
           'SOURCE INCONSISTENCY: printed root leaves +.685813mm equation residual')
    record('5.1',75,'suction_inlet_x',p45.suction.evaluate(0)[0]*1000,1.980,'mm',.0006)
    ca=.1/math.sin(math.radians(78))
    active_spec=ProfileSpec(.1,78,40,20,.58*ca,.53/25.61*ca,.2/25.61*ca)
    active_data=active_profile_construction(active_spec);active=build_active_profile(active_spec,variant='source')
    # Printed fully reconstructed circular source geometry, not corrected BCs.
    for name,value,expected in [('pressure_radius',active_data['pressure_radius_m']*1000,60.877),
                              ('pressure_center_x',active_data['pressure_center'][0]*1000,43.626)]:
        record('6.1',79,name,value,expected,'mm',.0006)
    export_csv(out/'control_errors.csv',rows);export_json(out/'control_errors.json',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(14,9))
    cases=[([0,0],[100,0],np.pi/3,-np.pi/6),([0,0],[100,10],2*np.pi/3,-np.pi/12),
           ([0,0],[100,-10],np.pi/2,-np.pi/4),([-10,10],[100,10],-np.pi/4,-np.pi/2)]
    for a,b,aa,bb in cases:
        q=parabolic_envelope(np.array(a)*.001,np.array(b)*.001,aa,bb).sample()
        axs[0,0].plot(q[:,0]*1000,q[:,1]*1000)
    axs[0,0].set_title('Table 2.2: all four parabolas')
    for ax,p,title in zip([axs[0,1],axs[0,2],axs[1,0]],[p31,p45,active],['Table3.1','Tables4.1/5.1','Table6.1 SOURCE: inlet BC invalid']):
        q=p.contour();ax.plot(q[:,0]*1000,q[:,1]*1000);ax.set_title(title)
    ax=axs[1,1]
    for shift in (0,c9017_spec.pitch_m):
        q=c9017.contour()+[0,shift];ax.plot(q[:,0]*1000,q[:,1]*1000)
    from matplotlib.patches import Circle
    for circle in channel.circles[::4]:ax.add_patch(Circle(circle.center_m*1000,circle.radius_m*1000,fill=False,lw=.6))
    ax.set_title('Table3.4: actual tangent circles')
    ax=axs[1,2]
    for q in (triangles.inlet_plot_m_s,triangles.outlet_plot_m_s):ax.plot(q[:,0],q[:,1],'o-')
    ax.set_title('Table3.5: signed triangle matrices')
    for ax in axs.flat:ax.axis('equal');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'source_controls.png',dpi=160);plt.close(fig)
    export_dxf(out/'source_profiles.dxf',[p31,p45,active],[p31_spec.pitch_m,p45_spec.pitch_m,active_spec.pitch_m])
    fig,ax=plt.subplots(figsize=(10,7))
    for curve in (*layout.stator_contours_m,*layout.rotor_contours_m):
        ax.plot(curve[:,0]*1000,curve[:,1]*1000)
    for curve in (layout.inlet_triangle_m,layout.outlet_triangle_m):
        ax.plot(curve[:,0]*1000,curve[:,1]*1000,'k-o',ms=3)
    ax.set(title='Table3.6: sequential Mathcad assignments; same source offsets',xlabel='Axial, mm',ylabel='Circumferential, mm')
    ax.axis('equal');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'elementary_stage_source.png',dpi=160);plt.close(fig)
    return dict(comparisons=rows,source_comparison_failures=[r for r in rows if not r['within_printed_tolerance']],
                figure=str(out/'source_controls.png'),status='EXPERIMENTAL; documented source discrepancies are not silently accepted')
