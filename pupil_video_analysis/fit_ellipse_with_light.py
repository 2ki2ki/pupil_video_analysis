from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import pandas as pd
from nptyping import NDArray
from pandas.core.common import flatten
from skimage.measure import EllipseModel
import glob

XCenter = float
YCenter = float
PupilLikelihood =float
EyeLikelihood =float
VLen = float
HLen = float
EventLightOn = float
Likelihood = float


A = float
B = float
Theta = float
EllipseParmas = Tuple[XCenter, YCenter, A, B, Theta]


Area = float

Color = Tuple[int, int, int]


Model = str
Bodyparts = str
Coordinate = str
DLCKey = Tuple[Model, Bodyparts, Coordinate]


def fit_ellipse(points: NDArray[2, float]) -> EllipseParmas:
    points = points[~np.isnan(points).any(axis=1), :]
    m = EllipseModel()
    m.estimate(points)
    return tuple(m.params)


def calc_ellipse_area(params: EllipseParmas) -> float:
    _, _, a, b, _ = params
    return np.pi * a * b


def draw_ellipse(frame: NDArray, params: EllipseParmas, color: Color,
                 thickness: int):
    xc, yc, a, b, theta = params
    angle = 180. * theta / np.pi
    cv2.ellipse(frame, ((xc, yc), (2 * a, 2 * b), angle),
                color,
                thickness=thickness)


def is_coordinate(key: DLCKey):
    return "x" in key or "y" in key


def extract_key_of_bodyparts(data: pd.DataFrame,
                             bodyparts: Bodyparts) -> List[DLCKey]:
    coords = list(filter(is_coordinate, data.keys()))
    return list(filter(lambda key: bodyparts in key[0], coords))



def reshape2fittable(data: pd.DataFrame) -> NDArray[3, float]:
    nrow, ncol = data.shape
    return np.array(data).reshape(nrow, -1, 2)


def as_output_filename(video_path: Path):
    parent = video_path.parents[1]
    filepath_without_extension = parent.joinpath("area").joinpath(
        video_path.stem)
    return str(filepath_without_extension) + ".csv"


def is_marked(frame: NDArray,likelihood: float, threshold: float,
            position: Tuple[int, int],color_range: Tuple[Color, Color]) -> int:
    y , x = position
    color = frame[x, y, :]
    lcolor, ucolor = color_range
    for comp in zip(color, lcolor, ucolor):
        c, l, u = comp
        if not (l <= c and c <= u):
            return 0
        if likelihood < threshold:
            return 0
    return 1


def to_video_path(h5path: Path) -> str:
    stem = h5path.stem.split("_interpolated")[0]
    return str(h5path.parents[0].joinpath("videos").joinpath(stem)) + ".MP4"

#Niki add 

def draw_circle(frame: NDArray, params, radius, color: Color,
                 thickness: int):
    xc, yc = params
    r = radius
    #print(xc,yc,r)
    cv2.circle(frame, (xc, yc),r,color,thickness=thickness)


def how_likelihood(key: DLCKey):
    return  "likelihood" in key 


def extract_key_of_event(data: pd.DataFrame,
                             bodyparts: Bodyparts) -> List[DLCKey]:
    coords = list(filter(how_likelihood, data.keys()))
    return list(filter(lambda key: bodyparts in key[0], coords))


def pickup_event(data: pd.DataFrame) -> NDArray[3, float]:
    nrow, ncol = data.shape
    return np.array(data).reshape(nrow, -1, 1)

def event_light_position(data,frame):
    data_list = data.flatten().tolist()
    data_list = list(map(float,data_list))
    data_list = np.array(data_list) -1 

    if data_list[0] >= frame.shape[1]:
        data_list[0] = (frame.shape[1]-1)

    if data_list[1] >= frame.shape[0]:
        data_list[1] = (frame.shape[0] -1 )

    return list(data_list)




    
if __name__ == '__main__':

    create_video = True
    show_video = True
    h5p = "C:/Users/Koji/analysis/pupil_video_analysis/pupil_video_analysis/videos_h5/interpolated_data/"
    h5s = glob.glob(h5p+"*interpolated.h5")


    for h5 in h5s:
        video = to_video_path(Path(h5))
        print(f"start processing {video}")

        tracked_data = pd.read_hdf(h5)
        cap = cv2.VideoCapture(str(video))
        nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if create_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            writer = cv2.VideoWriter(
                f"{h5p}elipse_video/{Path(video).stem}-ellipse.MP4", fourcc, fps,
                (width, height))

        pupil_keys = extract_key_of_bodyparts(tracked_data, "pupil")
        eyelid_keys = extract_key_of_bodyparts(tracked_data, "eyelid")

        pupil_data = reshape2fittable(tracked_data[pupil_keys])
        eyelid_data = reshape2fittable(tracked_data[eyelid_keys])

        nose_keys = extract_key_of_bodyparts(tracked_data, "nose")
        nose_data = reshape2fittable(tracked_data[nose_keys])
        

        FT_keys = extract_key_of_bodyparts(tracked_data, "FT")
        peak_keys = extract_key_of_bodyparts(tracked_data, "peak")
        FT_data = reshape2fittable(tracked_data[FT_keys])
        peak_data = reshape2fittable(tracked_data[peak_keys])

        FT_likelihood_keys = extract_key_of_event(tracked_data, "FT")
        peak_likelihood_keys = extract_key_of_event(tracked_data, "peak")
        FT_likelihood_data = pickup_event(tracked_data[FT_likelihood_keys])
        peak_likelihood_data = pickup_event(tracked_data[peak_likelihood_keys])


        results: List[Tuple[Area,VLen,HLen,Area,VLen,HLen,XCenter,YCenter,XCenter,YCenter,EventLightOn,Likelihood, EventLightOn,Likelihood]] = []
        for i in range(nframe):
            if i % 5000 == 0:
                print(f"Processing {i}-th frame")

            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame =cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            pupil_params = fit_ellipse(pupil_data[i])
            eyelid_params = fit_ellipse(eyelid_data[i])
     
            pupil_area = calc_ellipse_area(pupil_params)
            eyelid_area = calc_ellipse_area(eyelid_params)

            pupil_x, pupil_y, pupil_v_len, pupil_h_len, _ = pupil_params
            _, _, eye_v_len, eye_h_len, _ = eyelid_params

            nose_x,nose_y = event_light_position(nose_data[i],frame)

            FT_position = list(map(int,event_light_position(FT_data[i],frame)))
            peak_position =  list(map(int,event_light_position(peak_data[i],frame)))

            FT_likelihood = FT_likelihood_data.flatten()[i]
            peak_likelihood = peak_likelihood_data.flatten()[i]
            
            FT_on = is_marked(frame,FT_likelihood, 0.1, FT_position, ((225, 225, 225), (255, 255, 255)))
            peak_on = is_marked(frame,peak_likelihood, 0.05, peak_position, ((225, 225, 225), (255, 255, 255)))

            results.append((pupil_area,pupil_v_len, pupil_h_len, eyelid_area, eye_v_len,eye_h_len, pupil_x, pupil_y,nose_x,nose_y,FT_on,FT_likelihood,peak_on,peak_likelihood))

            if show_video:
                
                nose_draw_position = [int(nose_x),int(nose_y)]

                draw_ellipse(frame, pupil_params, (0, 0, 255), 1)
                draw_ellipse(frame, eyelid_params, (0, 255, 0), 1)
                draw_circle(frame,nose_draw_position,5, (255,0,0), -1)

                if FT_likelihood > 0.1:
                    draw_circle(frame, FT_position, 10, (0, 0, 255), -1)
                if peak_likelihood > 0.1:
                    draw_circle(frame, peak_position, 10, (255, 0, 0), -1)
                if FT_on == 1:
                    draw_circle(frame, FT_position, 10, (0, 0,255), 1)
                if peak_on == 1:
                    draw_circle(frame, peak_position, 10, (255, 0, 0), 1)

            if create_video:
                writer.write(frame)

            if show_video:
                cv2.imshow("video", frame)

                if cv2.waitKey(1) % 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    cap.release()
                    break


        if create_video:
            writer.release()
        output = pd.DataFrame(
            results,
            columns=["pupil_area","pupil_v_len","pupil_h_len","eyelid_area", "eye_v_len","eye_h_len","pupil_center_x", "pupil_center_y","nose_x","nose_y","FT_on", "FT_likelihood","peak_on","peak_likelihood"])
        output_path = as_output_filename(Path(video))
        output.to_csv(output_path, index=False)

