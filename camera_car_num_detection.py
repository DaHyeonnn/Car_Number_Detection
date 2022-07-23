from re import A, T
from wsgiref.util import request_uri
import cv2
import pyrealsense2 as rs
import numpy as np
import pytesseract
import matplotlib.pyplot as plt


MIN_AREA = 80  # bounding 최소 넓이 (원래 코드는 80)
MAX_AREA = 1000
MIN_WIDTH, MIN_HEIGHT = 2, 8    # 너비 높이
MIN_RATIO, MAX_RATIO = 0.25, 1.0    # 가로:세로 비율 -> 번호판 글자가 1:4 비율정도 되기 때문에 이렇게 설정해줌

plt.style.use('dark_background')

def find_chars(contour_list):
    MAX_DIAG_MULTI = 5 # 번호 사이의 길이
    MAX_ANGLE_DIFF = 12.0 # 번호와 번호 사이의 각도 
    MAX_AREA_DIFF = 0.5 # 번호마다 면적의 차이 -> 면적차이가 작아야 같은 글자로 인식
    MAX_W_DIFF = 0.8
    MAX_H_DIFF = 0.2
    MIN_N_MATCHED = 3   #이어진 번호 개수 -> 박스가 2개 이상이면 번호로 치기

    matched_result_idx = [] # 최종 인덱스 저장 리스트
    
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if distance < diagonal_length1 * MAX_DIAG_MULTI \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_W_DIFF and height_diff < MAX_H_DIFF:
                matched_contours_idx.append(d2['idx'])

     
        matched_contours_idx.append(d1['idx']) # 인덱스 추가

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
        
        # 재귀
        recursive_contour_list = find_chars(unmatched_contour)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx
    

# # # draw 함수 # # #
###################################################################################3
def draw(images, height, width, channel) :
    possible_contours = []      #  contours_dict 리스트 데이터에서 다시 뽑아낼 데이터 리스트 생성
    cnt = 0
    img_blur = cv2.GaussianBlur(images, ksize=(5, 5), sigmaX=0)
        # 블러 처리, 노이즈 제거를 위함 GaussianBlur(이미지, 필터 크기, 표준편차)
    
    img_blur_thresh = cv2.adaptiveThreshold(    #이미지 전처리 과정, 이진화
        img_blur, 
        maxValue=255.0, #기준값을 넘었을 때, 255로 적용
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # 기준값을 계산하는 방법
        thresholdType=cv2.THRESH_BINARY_INV, # 임계 처리 유형
        blockSize=19,  # block bx 영역 크기
        C=15 # 평균에서 차감할 값
        #C 값을 크게 줄수록 BOX 개수가 적어짐o
    ) 
    
    contours,_ = cv2.findContours(
        img_blur_thresh, 
        mode=cv2.RETR_LIST, # 외곽선 검출 모드 
        method=cv2.CHAIN_APPROX_SIMPLE # 외곽선 근사화 방법
    ) 

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
    #(입력데이터, 외곽선 좌표정보, 외곽선 인덱스 (-1은 모든 외각선), 색상
    contours_dict = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
    # 찾은 윤곽선을 감싸는 사각형 bounding box 반환 -> x,y,w,h 좌표
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 255), thickness=2)
    # (데이터, 시작점 좌표, 종료점 좌표, 색상, 박스 두께)

        contours_dict.append({
            'contour': contour, # 윤곽선
            'x': x, 'y': y,'w': w,'h': h,     # x,y,w,h 좌표
            'cx': x + (w / 2), 'cy': y + (h / 2) # bounding box 좌표의 중심값
        })
    temp_result = cv2.cvtColor(temp_result, cv2.COLOR_BGR2GRAY)
    
    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

    for d in contours_dict:
        area = d['w'] * d['h']  # 넓이 = 가로 * 세로
        ratio = d['w'] / d['h'] # 비율 = 가로 / 세로
    
        if MIN_AREA < area < MAX_AREA \
        and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)
        

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
  
    return temp_result, possible_contours, img_blur_thresh


def min_box(possible_contours, height, width, channel):

    result_idx = find_chars(possible_contours) #찾은 문자열 저장

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)     

    return temp_result, matched_result

plate_imgs = []
plate_infos = []

def crop(matched_result):
    PLATE_WIDTH_PADDING = 1.3 # 1.3
    PLATE_HEIGHT_PADDING = 1.5 # 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10


    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )

        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

        img_rotated = cv2.warpAffine(img_blur_thresh, M=rotation_matrix, dsize=(width, height))

        img_cropped = cv2.getRectSubPix(
            img_rotated, 
            patchSize=(int(plate_width), int(plate_height)), 
            center=(int(plate_cx), int(plate_cy))
            )

        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue

    
    plate_imgs.append(img_cropped)
    plate_infos.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })
        

    for i, plate_img in enumerate(plate_imgs):
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # find contours again (same as above)
        contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        
        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            area = w * h
            ratio = w / h

            if area > MIN_AREA \
            and w > MIN_WIDTH and h > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
                if x < plate_min_x:
                    plate_min_x = x
                if y < plate_min_y:
                    plate_min_y = y
                if x + w > plate_max_x:
                    plate_max_x = x + w
                if y + h > plate_max_y:
                    plate_max_y = y + h
                    
        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
        
        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
        _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
   
        chars = pytesseract.image_to_string(img_result , lang='kor')
        
    return img_result, chars


if __name__ == "__main__":


# Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.bgr8, 30)

    if device_product_line != 'L500':
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        print('ok')

    # Start streaming
    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 카메라 == images
            images = np.asanyarray(color_frame.get_data())
            
            cv2.imshow('RealSense', images)
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            height, width, channel = images.shape
            result, possible_contours, img_blur_thresh = draw(gray, height, width, channel)
            cv2.imshow('bounding box', result)
            result2, matched_result = min_box(possible_contours, height, width, channel)
        
            if len(matched_result) != 0:
                result3 , chars = crop(matched_result)  
                res = cv2.addWeighted(images, 0.8, result, 0.5, 0.)
                cv2.imshow("car number find", res)

                chars = pytesseract.image_to_string(result3, lang='kor')

                for i in chars:
                    if ord('가') <= ord(i) <= ord('힣'):
                        n = 1
                        break
                    else : n = 0

                if (n == 1) and len(chars) > 9:
                    print("번호판 : "+chars)
                
                cv2.imshow('final crop', result3)
                    
            else: 
                pass

            if cv2.waitKey(100) &0xFF == ord('q'):
                break

    finally:

        # Stop streaming
        pipeline.stop()
