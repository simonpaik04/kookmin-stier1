# vision_pkg_ros2 소스 코드 다이어그램

이 문서는 `vision_pkg_ros2`의 실제 Python 소스 코드 기준으로 프레임 처리 순서와
주요 클래스 의존 관계를 Mermaid Markdown으로 정리합니다.

## 1. lane_node 프레임 처리 시퀀스

```mermaid
sequenceDiagram
    autonumber
    participant Camera as Camera Topic
    participant LaneNode as LaneNode
    participant Bridge as CvBridge
    participant Processor as LaneProcessor
    participant Line as LineProcessor
    participant Traffic as TrafficLightProcessor
    participant Visualizer as VisionVisualizer
    participant Publishers as ROS Publishers

    Camera->>LaneNode: sensor_msgs/Image or CompressedImage
    LaneNode->>LaneNode: TARGET_FPS gate
    alt frame interval too short
        LaneNode-->>Camera: return without publish
    else process frame
        LaneNode->>Bridge: convert ROS image to OpenCV BGR
        alt image conversion failed
            Bridge-->>LaneNode: exception
            LaneNode-->>Camera: log error and return
        else conversion ok
            Bridge-->>LaneNode: frame
            LaneNode->>Processor: process(frame)
            Processor->>Processor: undistort if mode == real
            Processor->>Line: process(frame)
            Line-->>Processor: lane mask, BEV, stopline, lane objects, schoolzone
            Processor->>Traffic: process(raw_frame)
            Traffic-->>Processor: detections, traffic_light_status, traffic_decide
            Processor->>Visualizer: build_images(line, traffic, bev, raw_frame)
            Visualizer-->>Processor: final_result, front_yolo
            Processor-->>LaneNode: result dict
            LaneNode->>Publishers: /vision/lane_mask
            LaneNode->>Publishers: /vision/bev
            LaneNode->>Publishers: /vision/final_result
            LaneNode->>Publishers: /vision/front_yolo
            LaneNode->>Publishers: /vision_objs
            LaneNode->>Publishers: /schoolzone
            LaneNode->>Publishers: /stoplane
            LaneNode->>Publishers: /vision/stopline_distance
            LaneNode->>Publishers: /trafficlight
        end
    end
```

![lane_node 프레임 처리 시퀀스](source_code_diagram_images/lane_node_sequence.png)

## 2. 차선·정지선·스쿨존 처리 시퀀스

```mermaid
sequenceDiagram
    autonumber
    participant LaneProcessor as LaneProcessor
    participant LineProcessor as LineProcessor
    participant BevProcessor as BevProcessor
    participant Projector as FrontBevProjector
    participant VisionUtils as vision_utils
    participant Clusterer as LaneObjectClusterer
    participant Schoolzone as SchoolzoneDetector

    LaneProcessor->>LineProcessor: process(frame)
    LineProcessor->>BevProcessor: warp(frame)
    BevProcessor->>Projector: warp(frame)
    Projector-->>BevProcessor: BEV color image
    BevProcessor-->>LineProcessor: BEV color image
    LineProcessor->>VisionUtils: keep_lane_colors(bev_color)
    LineProcessor->>VisionUtils: find_largest_black_box(bev_color)
    VisionUtils-->>LineProcessor: start_black_box or None
    LineProcessor->>VisionUtils: black_out_box(bev_img, start_black_box)
    LineProcessor->>VisionUtils: split_white_yellow_masks(bev_img)
    VisionUtils-->>LineProcessor: white_mask, yellow_mask
    LineProcessor->>VisionUtils: detect_stop_line(white_mask)
    VisionUtils-->>LineProcessor: stop_line or None
    LineProcessor->>VisionUtils: stop_line_distance_m(stop_line, height)
    LineProcessor->>VisionUtils: remove_stop_line_from_masks(stop_line, masks)
    VisionUtils-->>LineProcessor: lane_white_mask, lane_yellow_mask
    LineProcessor->>Clusterer: build(object_white_mask, object_yellow_mask)
    Clusterer-->>LineProcessor: lane_objects with label 2 or 3
    LineProcessor->>Schoolzone: detect(lane_objects)
    Schoolzone-->>LineProcessor: Bool
    LineProcessor-->>LaneProcessor: line result dict
```

![차선 정지선 스쿨존 처리 시퀀스](source_code_diagram_images/line_pipeline_sequence.png)

## 3. 신호등 처리 시퀀스

```mermaid
sequenceDiagram
    autonumber
    participant LaneProcessor as LaneProcessor
    participant TrafficProcessor as TrafficLightProcessor
    participant Crop as traffic_light_crop
    participant Yolo as YoloNet
    participant Classifier as TrafficLightColorClassifier
    participant Decider as TrafficLightDecider

    LaneProcessor->>TrafficProcessor: process(raw_frame)
    TrafficProcessor->>TrafficProcessor: check YOLO_ENABLED and frame skip
    alt YOLO disabled or skipped
        TrafficProcessor-->>LaneProcessor: cached or empty detections
    else run YOLO
        TrafficProcessor->>Yolo: load()
        alt model load failed
            Yolo-->>TrafficProcessor: None
            TrafficProcessor-->>LaneProcessor: traffic_decide = UNKNOWN
        else model ready
            TrafficProcessor->>Crop: crop_top(frame)
            Crop-->>TrafficProcessor: top crop
            TrafficProcessor->>Yolo: predict(crop, imgsz, conf, classes)
            Yolo-->>TrafficProcessor: YOLO results
            loop each traffic-light bbox
                TrafficProcessor->>Classifier: classify(frame, bbox)
                Classifier-->>TrafficProcessor: state, state_en, color
            end
            TrafficProcessor->>Decider: select_status(detections)
            Decider-->>TrafficProcessor: status
            TrafficProcessor->>Decider: status_to_decision(status)
            Decider-->>TrafficProcessor: Int32 decision
            TrafficProcessor-->>LaneProcessor: traffic result dict
        end
    end
```

![신호등 처리 시퀀스](source_code_diagram_images/traffic_light_sequence.png)

## 4. 주요 클래스 다이어그램

```mermaid
classDiagram
    class LaneNode {
      +callback(msg)
      +main()
      -_declare_parameters()
      -_load_parameters()
      -_apply_tunable_parameters()
      -_on_parameter_update(params)
    }

    class LaneProcessor {
      +process(frame) dict
      +rebuild_bev_projector()
      -_undistort_if_needed(frame)
      -_log_performance(...)
    }

    class LineProcessor {
      +process(frame) dict
      +rebuild_bev()
      -_remove_center_yellow_for_lane_mask(mask)
    }

    class BevProcessor {
      +warp(frame)
      +draw_vehicle(image)
      +rebuild()
      -_sync_config_shape()
    }

    class FrontBevProjector {
      +warp(image)
      +draw_vehicle(image)
      -_build_warp_lut(homography)
    }

    class LaneObjectClusterer {
      +build(white_mask, yellow_mask)
      +bev_pixel_to_local_m(px, py, h, w)
      -_build_from_mask(mask, label, max_count)
      -_estimate_yaw(xs, ys, h, w)
    }

    class SchoolzoneDetector {
      +detect(lane_objects) bool
    }

    class TrafficLightProcessor {
      +process(frame) dict
      -_run_yolo(frame)
      -_detect_traffic_lights_top_crop(frame)
      -_traffic_light_class_ids()
      -_clear_yolo_state()
    }

    class YoloNet {
      +load()
      +predict(image, imgsz, conf, classes)
    }

    class TrafficLightColorClassifier {
      +classify(frame, bbox) dict
      +mask_stats(mask)
    }

    class TrafficLightDecider {
      +select_status(detections)
      +status_to_decision(status)
      +status_to_ascii(status)
      +normalize_yolo_name(name)
      +is_traffic_light_name(name)
      +traffic_light_box_score(det)
    }

    class VisionVisualizer {
      +build_images(line, traffic, bev_processor, raw_frame)
      +draw_lane_objects(image, lane_objects)
      +draw_yolo_overlay(final_img, detections, status)
      +draw_yolo_boxes(image, detections, status)
      +draw_traffic_light_status(image, status)
      +draw_schoolzone_status(image, detected)
      +draw_stopline_status(image, distance_m, stopline_detected)
    }

    class RosMessageUtils {
      <<module>>
      +create_lane_objects_msg(lane_objects, stamp, frame_id)
    }

    class VisionUtils {
      <<module>>
      +keep_lane_colors(image)
      +find_largest_black_box(image)
      +black_out_box(image, box)
      +split_white_yellow_masks(image)
      +detect_stop_line(mask)
      +stop_line_distance_m(stop_line, height)
      +remove_stop_line_from_masks(stop_line, masks)
    }

    LaneNode *-- LaneProcessor
    LaneNode --> RosMessageUtils
    LaneProcessor *-- LineProcessor
    LaneProcessor *-- TrafficLightProcessor
    LaneProcessor *-- VisionVisualizer
    LineProcessor *-- BevProcessor
    LineProcessor *-- LaneObjectClusterer
    LineProcessor *-- SchoolzoneDetector
    LineProcessor --> VisionUtils
    BevProcessor *-- FrontBevProjector
    TrafficLightProcessor *-- YoloNet
    TrafficLightProcessor *-- TrafficLightColorClassifier
    TrafficLightProcessor --> TrafficLightDecider
    VisionVisualizer --> TrafficLightDecider
```

![vision_pkg_ros2 주요 클래스 다이어그램](source_code_diagram_images/main_class_diagram.png)

## 5. 선택 실행 도구

```mermaid
classDiagram
    class TrafficLightYoloViewer {
      -_on_image(msg)
      +_color_for_state(state)
    }

    class TrafficLightDatasetCapture {
      -_on_image(msg)
      -_write_dataset_yaml()
    }

    class DatasetFiles {
      <<artifact>>
      images
      labels
      data.yaml
    }

    class YoloNet {
      +load()
      +predict(image, imgsz, conf, classes)
    }

    TrafficLightYoloViewer --> YoloNet : direct YOLO preview
    TrafficLightDatasetCapture --> DatasetFiles : write
```

![비전 보조 도구 클래스 다이어그램](source_code_diagram_images/tool_class_diagram.png)

`traffic_light_yolo_viewer`와 `traffic_light_dataset_capture`는 `vision.launch.py`에서 자동 실행되지 않습니다.
필요할 때 별도 터미널에서 실행하는 보조 도구입니다.
