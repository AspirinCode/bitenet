from bitenet import *
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("pdb", help="path to pdb file")
parser.add_argument("out", help="path to out image file")
parser.add_argument("--rotations", help="if set, additional rotations are added", action="store_true")
parser.add_argument("--score_threshold", help="score threshold for predictions", type=float, default=-1)
args = parser.parse_args()

# load pdb
cmd.load(args.pdb, object="protein")
if "5d41" in args.pdb:
    view = \
        [0.315078378,    0.941709995,    0.117241584,\
         0.053722709,   -0.141038641,    0.988448024,\
         0.947452068,   -0.305185378,   -0.095028281,\
         0.000000000,    0.000000000, -280.114013672,\
       -38.076297760,   44.115356445,   -5.016201019,\
       -45.282546997,  605.510559082,  -20.000000000]
elif "5yve" in args.pdb:
    view = \
        [0.992645681,   -0.121056922,    0.000000000,\
         0.121056922,    0.992645681,    0.000000000,\
         0.000000000,    0.000000000,    1.000000000,\
         0.000000345,    0.000001714, -243.703536987,\
         0.365299344,    7.096632004,  -46.994018555,\
       162.740753174,  324.666229248,  -20.000000000]
else:
    view = cmd.get_view()
cmd.remove("solvent")
cmd.color(color_names_all[0], "protein and polymer.protein and name C*")

# init model
model = BiteNet_Draw()
model.working_dir = "./data/"

model.prediction_drawer.radius = 2

model.dataloader.rotation_eval = args.rotations
if args.score_threshold > 0:
    model.prediction_processer.score_threshold = args.score_threshold

# run prediction
model("protein")

cmd.set("ray_shadows", 0)
cmd.set("cartoon_oval_length", 0.8)
cmd.set("cartoon_rect_length", 0.8)
cmd.set("sphere_scale", 0.7)
cmd.set("orthoscopic", True)
cmd.bg_color("white")
cmd.set("opaque_background", 0)
cmd.set("label_size", 28)
cmd.set("label_color", "black")
cmd.set("label_outline_color", "white")
cmd.set("label_position", [2, 2, 10])
cmd.set_view(view)
# make image
cmd.png(args.out, width=720, height=720, dpi=-1, ray=1, quiet=1)

cmd.quit()