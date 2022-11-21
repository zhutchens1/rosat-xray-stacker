import pandas as pd
from rosat_xray_stacker import rosat_xray_stacker

"""
This file does the X-ray stacking for RESOLVE-B.
"""


if __name__=='__main__':
    g3groups = pd.read_csv("../g3groups/RESOLVE_G3groupcatalog_052821.csv")
    g3groups = g3groups[(g3groups.g3fc_l==1)] # select only centrals since we want group info (one central per group)

    rosat_xray_stacker('/srv/two/zhutchen/rosat_xray_stacker/g3rassimages/resb/',\
                       g3groups.g3grpradeg_l, g3groups.g3grpdedeg_l, g3groups.g3grp_l, \
                       surveys=['RASS-Int Broad', 'RASS-Int Soft', 'RASS-Int Hard'],\
                       centralname=g3groups.name)
