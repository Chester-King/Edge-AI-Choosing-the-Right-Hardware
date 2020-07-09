
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords,w_scale):
        
        '''
        There is no need for h_scale because of obvious reasons :)
        '''
        
        d={k+1:0 for k in range(len(self.queues))}
        
        for coord in coords:
            for i, q in enumerate(self.queues):

                tw=int(coord[3]*w_scale)
                th=int(coord[5]*w_scale)
#                 print(tw,'-',th,'---',q)
                if tw>q[0] and th<q[2]:
                    d[i+1]+=1
        print('Count Status : ',d)
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        
        self.infer_request_handle = None

        try:
#             self.model=IECore.read_network(self.model_structure, self.model_weights)
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        
        # Loading Up plugin and network
        self.plugin = IECore()
        self.net_plugin = self.plugin.load_network(network=self.model, device_name=self.device, num_requests=1)
        print('Hello_load')
                
#         raise NotImplementedError
        
    def predict(self, image):
        
        
        
        infer_request_handle = self.net_plugin.start_async(request_id=0, inputs={self.input_name: self.preprocess_input(image)})
        if(infer_request_handle.wait()==0):
            net_output = infer_request_handle.outputs[self.output_name]
        print('Hello_predict')
        return(net_output,image)
    
#         raise NotImplementedError
    
    def draw_outputs(self, coords, image,initial_w,initial_h):
        print('Hello_DO')
        c_count = 0
        de_obj=[]
        for x in coords[0][0]:
            if(x[2] >= self.threshold):
                    xmin = int(x[3] * initial_w)
                    ymin = int(x[4] * initial_h)
                    xmax = int(x[5] * initial_w)
                    ymax = int(x[6] * initial_h)
                    cv2.rectangle(image, (xmin, ymin),(xmax, ymax), (0, 0, 255), 1)
                    c_count += 1
                    
                    de_obj.append(x)
        
        return(image, c_count, de_obj)
    
#         raise NotImplementedError

    def preprocess_outputs(self, outputs):
        
        '''
        This function is not used anywhere tho... don't know why it is here but gotta complete so ¯\_(ツ)_/¯
        '''
        print('Hello Pre_O')
        
        
#         raise NotImplementedError

    def preprocess_input(self, image):
        n,c,h,w = self.input_shape
        im_frame = cv2.resize(image, (w, h),interpolation = cv2.INTER_AREA)
        im_frame = im_frame.transpose((2, 0, 1))
        im_frame = im_frame.reshape((n, c, h, w))
        return(im_frame)
#         raise NotImplementedError


def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    start_inference_time=time.time()

#     try:
    while cap.isOpened():
        ret, frame=cap.read()
        if not ret:
            break
        counter+=1        
        coords, image= pd.predict(frame)
        frame, current_count, coords = pd.draw_outputs(coords, image,initial_w,initial_h)
        num_people= queue.check_coords(coords,initial_w)
        print(f"Total People in frame = {len(coords)}")
        print(f"Number of people in queue = {num_people}")
        out_text=""
        y_pixel=25          
        for k, v in num_people.items():
            out_text += f"No. of People in Queue {k} is {v} "
            if v >= int(max_people):
                out_text += f" Queue full; Please move to next Queue "
            cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            out_text=""
            y_pixel+=40
        out_video.write(image)
            
    total_time=time.time()-start_inference_time
    total_inference_time=round(total_time, 1)
    fps=counter/total_inference_time

    with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
        f.write(str(total_inference_time)+'\n')
        f.write(str(fps)+'\n')
        f.write(str(total_model_load_time)+'\n')

    cap.release()
    cv2.destroyAllWindows()
#     except Exception as e:
#         print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)