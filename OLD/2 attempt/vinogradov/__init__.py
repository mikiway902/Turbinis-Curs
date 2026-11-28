"""Experimental Vinogradov (2010) blade-design supplement to Mikhaltsev.

Importing does not run optimization, modify globals or change legacy results.
See docs/vinogradov/SOURCE_MAP.md for equation status and source provenance.
"""
from .models import ConstructionError, FlowStation, ProfileSpec, RadialSectionInput, SectionProfile
from .parabolic_profile import build_parabolic_profile, build_scheme, parabolic_envelope
from .double_parabolic_profile import build_prescribed_thickness_profile, build_double_parabolic_profile
from .active_profile import build_active_profile
from .integration import design_from_namespace

__all__ = ['ConstructionError','FlowStation','ProfileSpec','RadialSectionInput','SectionProfile',
           'build_parabolic_profile','build_scheme','parabolic_envelope',
           'build_prescribed_thickness_profile','build_double_parabolic_profile','build_active_profile',
           'design_from_namespace']
