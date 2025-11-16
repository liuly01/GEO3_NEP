import arcpy
from arcpy.sa import *
import os
import glob

outputpath = "xls/fluxcom-xbase"

shp = "Figure11/RECCAP_REGION.shp"
zoneField = "gridcode"

globaldir = "yearly/NEP"
NEE_global = glob.glob(os.path.join(globaldir, "*.tif"))
for dataname in NEE_global:
    filename = os.path.splitext(os.path.basename(dataname))[0]
    print(filename)
    outTable = (outputpath + "/" + filename +"_reccap.dbf")
    outexcel = (outputpath + "/" + filename +"_reccap.xls")
    outZSaT = ZonalStatisticsAsTable(shp, zoneField, dataname, outTable, "DATA", "SUM")
    arcpy.TableToExcel_conversion(outTable, outexcel)

print('finish')

