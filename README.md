In this project I changed ultralytics for training with arbitrary reg_max. It is important to change the reg_max parameter in the scripts train_v2, ~/ultralytics/ultralytics/nn/modules/head.py , ~/ultralytics/ultralytics/ultralytics/utils/tal.py. 
The custom configuration for yolov11m-seg is also used(the only change is in the last line - [[16, 19, 22], 1, Segment, [nc, 128, 256]]  # Detect(P3, P4, P5)
 For example, if reg_max=128 )
The parameter is responsible for the dfl branch in the segmentation head. Details:  
https://zhuanlan.zhihu.com/p/13303751794#:~:text=%23%20post%2Dprocess%0A%20%20%20%20bboxes%2C%20scores%2C%20labels%20%3D%20post_process_det(%0A%20%20%20%20%20%20%20%20score_preds%2C%20xyxy_preds%2C%20conf_thresh%3D0.2%2C%20nms_thresh%3D0.5%2C%20num_classes%3D80)
