"""Real CSV/JSON/DXF/STEP exports; never fake STEP text or zero-volume shells.

STEP uses OpenCascade via optional cadquery. Discrete ruled loft preserves
input stations (no smoothing); tessellation accuracy is explicitly recorded.
DXF carries exact 2D Bezier control polygons and circle arcs in millimetres.
"""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from pathlib import Path
import csv
import json
import math
import numpy as np
from .curves import BezierCurve, CircularArc


def json_value(value):
    if is_dataclass(value):return json_value(asdict(value))
    if isinstance(value,np.ndarray):return json_value(value.tolist())
    if isinstance(value,np.generic):return json_value(value.item())
    if isinstance(value,dict):return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_value(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(f'unsupported JSON report type: {type(value).__name__}')


def export_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(json_value(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return path


def export_csv(path,rows):
    rows=list(rows)
    if not rows:raise ValueError('cannot export an empty result table')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(json_value(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple,np.ndarray)) else json_value(v) for k,v in row.items()})
    return path


def export_dxf(path,profiles,pitches_m=None):
    import ezdxf
    doc=ezdxf.new('R2010');doc.units=ezdxf.units.MM;msp=doc.modelspace()
    if pitches_m is None:pitches_m=[None]*len(profiles)
    for j,(profile,pitch) in enumerate(zip(profiles,pitches_m)):
        layer=f'SECTION_{j}';doc.layers.new(layer)
        offset=np.array([j*.2,0.])
        curves=(*profile.suction.segments,*profile.pressure.segments,profile.leading_edge,profile.trailing_edge)
        for displacement in ([0.] if pitch is None else [0.,pitch]):
            shift=offset+np.array([0,displacement])
            for curve in curves:
                if isinstance(curve,BezierCurve):
                    points=[(*((p+shift)*1000),0.) for p in curve.control_points]
                    if len(points)==2:msp.add_line(points[0],points[1],dxfattribs={'layer':layer})
                    else:
                        spline=msp.add_spline(degree=len(points)-1,dxfattribs={'layer':layer})
                        spline.set_open_uniform(points,degree=len(points)-1)
                elif isinstance(curve,CircularArc):
                    start,end=curve.start_angle,curve.end_angle
                    if end<start:start,end=end,start
                    msp.add_arc((curve.center+shift)*1000,curve.radius*1000,
                                math.degrees(start),math.degrees(end),dxfattribs={'layer':layer})
                else:raise TypeError('DXF exact export supports analytic Bezier and circular arcs only')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);doc.saveas(path)
    return path


def _cylindrical_loft(blade):
    """Sew a ruled side shell to TRUE cylindrical root/tip caps.

    A constant-radius section is not a planar 3D wire. UV edges below lie on
    their cylinder, so caps use actual radial surfaces, not invalid plane caps.
    """
    import cadquery as cq
    from OCP.gp import gp_Ax3,gp_Pnt,gp_Dir,gp_Pnt2d
    from OCP.Geom import Geom_CylindricalSurface
    from OCP.GCE2d import GCE2d_MakeSegment
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge,BRepBuilderAPI_MakeWire,
                                   BRepBuilderAPI_MakeFace,BRepBuilderAPI_Sewing,
                                   BRepBuilderAPI_MakeSolid)
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.BRepLib import BRepLib
    from OCP.TopoDS import TopoDS
    axis=gp_Ax3(gp_Pnt(0,0,0),gp_Dir(1,0,0),gp_Dir(0,1,0))
    loft=BRepOffsetAPI_ThruSections(False,True,1e-6)
    loft.CheckCompatibility(False)
    caps=[]
    for index,(r,p) in enumerate(zip(blade.radii_m,blade.contours_m)):
        surface=Geom_CylindricalSurface(axis,float(r)*1000)
        uv=np.column_stack((p[:,1]/r,p[:,0]*1000))
        wire=BRepBuilderAPI_MakeWire()
        for a,b in zip(uv[:-1],uv[1:]):
            if np.linalg.norm(a-b)<=1e-12:continue
            segment=GCE2d_MakeSegment(gp_Pnt2d(*a),gp_Pnt2d(*b)).Value()
            edge=BRepBuilderAPI_MakeEdge(segment,surface).Edge()
            if not BRepLib.BuildCurves3d_s(edge):
                raise ValueError('OpenCascade could not build cylindrical section edge')
            wire.Add(edge)
        if not wire.IsDone():raise ValueError('invalid cylindrical section wire')
        w=wire.Wire();loft.AddWire(w)
        if index in (0,len(blade.radii_m)-1):
            cap=BRepBuilderAPI_MakeFace(surface,w,True)
            if not cap.IsDone():raise ValueError('invalid cylindrical cap')
            caps.append(cap.Face())
    loft.Build()
    if not loft.IsDone():raise ValueError('ruled side-shell loft failed')
    sewing=BRepBuilderAPI_Sewing(1e-5)
    sewing.Add(loft.Shape());sewing.Add(caps[0]);sewing.Add(caps[-1]);sewing.Perform()
    shell=TopoDS.Shell_s(sewing.SewedShape())
    builder=BRepBuilderAPI_MakeSolid(shell)
    if not builder.IsDone():raise ValueError('cylindrical caps did not close the blade shell')
    solid=builder.Solid()
    if not BRepLib.OrientClosedSolid_s(solid):raise ValueError('blade shell is open or incoherent')
    return cq.Solid(solid)


def export_step(path,blades,counts,*,allow_unconverged=False):
    """Ruled solid blade lofts and complete periodic rows in a STEP assembly.

    Circumferential coordinates wrap at each actual section radius, not about
    an arbitrary local centroid. Input discretization uncertainty remains in
    companion JSON and must not be presented as exact continuous CAD geometry.
    """
    import cadquery as cq
    if len(blades)!=len(counts):raise ValueError('need a count for each row')
    assembly=cq.Assembly(name='turbine');volumes=[]
    for index,(blade,count) in enumerate(zip(blades,counts)):
        if not blade.mesh_converged and not allow_unconverged:
            raise ValueError('refusing unconverged radial CAD mesh')
        if not isinstance(count,int) or count<3:raise ValueError('invalid blade count')
        solid=_cylindrical_loft(blade)
        if not solid.isValid() or solid.Volume()<=0:raise ValueError('OpenCascade rejected lofted blade solid')
        volumes.append(float(solid.Volume())*1e-9)
        # Locations share one topology per row. rotate() transforms/copies an
        # entire dense loft for every blade, needlessly multiplying memory.
        for k in range(count):
            location=cq.Location(cq.Vector(0,0,0),cq.Vector(1,0,0),360*k/count)
            assembly.add(solid,name=f'row_{index}_blade_{k}',loc=location)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assembly.save(str(path),exportType='STEP',mode='default')
    return dict(path=str(path),blade_count=sum(counts),row_blade_volume_m3=volumes,
                cad_units='mm',representation='ruled side shell with cylindrical caps; polygonal UV contour, no smoothing')
