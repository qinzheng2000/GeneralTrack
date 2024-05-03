exp_file = 'exps/example/bdd100k/yolox_x.py'

# tracking args
track_thresh = 0.35
det_thresh = 0.45
track_buffer = 50
match_thresh = 0.9
min_box_area = 100
byte = True

# relation args
resize = [720, 1280]
restore_ckpt = 'checkpoints/generaltrack_bdd.pth'
roialign_size = (2, 2)
corr_radius = 4
corr_levels = 4