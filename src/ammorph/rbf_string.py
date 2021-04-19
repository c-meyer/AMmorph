
def gaussian_spline(epsilon):
    return r'exp(-(r*r) / (epsilon*epsilon))'.replace('epsilon', str(epsilon))


def multi_quadratic_biharmonic_spline(epsilon):
    return r'sqrt((r * r) + (epsilon * epsilon))'.replace('epsilon', str(epsilon))


def thin_plate_spline(epsilon):
    return r'where((r/epsilon) > 0, (r*r/(epsilon*epsilon))*log(r*r/(epsilon*epsilon)), r*r/(epsilon*epsilon))'.replace('epsilon', str(epsilon))


def beckert_wendland_c2_basis(epsilon):
    return r'where((1.0 - r/epsilon) > 0.0, (r/epsilon*4.0 + 1.0)*(1.0 - r/epsilon)**4, 0.0)'.replace('epsilon', str(epsilon))

