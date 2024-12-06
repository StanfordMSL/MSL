import numpy as np
from datetime import datetime
import torch
import os
from typing import List,Literal
import albumentations as A
from albumentations.pytorch import ToTensorV2
import imageio

class FlightRecorder():
    def __init__(self,
                 nx:int,nu:int,
                 hz:int,tf:float,
                 cam_dim:List[int],
                 obj:np.ndarray,
                 cohort_name:str,course_name:str,pilot_name:str,
                 Ntsol:int=5,Nimps:int=20):
        """
        FlightRecorder class for recording flight data and images. We record images every
        n_im time steps to save space.

        Args:
        nx:             Number of states
        nu:             Number of inputs
        hz:             Sampling frequency
        tf:             Final time
        cam_dim:        Dimensions of the camera image
        obj:            objective
        cohort_name:    Name of the cohort
        course_name:    Name of the course
        pilot_name:     Name of the pilot
        Ntsol:          Dimension of solve times vector
        Nimps:          Number of images per second

        Variables:
        Tact:           Actual time
        Uact:           Actual input
        Xref:           Reference state
        Uref:           Reference input
        Xest:           Estimated state
        Xext:           External state
        Adv:            Advisor values
        Tsol:           Solve times
        k:              Counter for the number of data points
        n_img:          Number of time steps between image recordings
        Imgs:           Image data
        process_image:  Process image function
        output_path:    Path to the output directory
        output_base:    Base name for the output file

        Returns:
        None

        """
        
        # Some useful path(s)
        workspace_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        # Check if output directory exists, if not, create it
        output_path = os.path.join(workspace_path,"cohorts",cohort_name,"flight_data")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Some useful constants
        Ndata = int(hz*tf+1)
        n_img = int(Ndata/(tf*Nimps))
        Nimgs = Ndata//n_img

        # Initialize the objective
        self.obj = obj

        # Initialize the image data
        self.n_img = n_img                              # Image step count
        self.Nimps = Nimps                              # Number of images per second
        self.Imgs  = np.zeros((Nimgs,cam_dim[0],cam_dim[1],cam_dim[2]))

        # Initialize the trajectory data
        self.Tact = np.zeros(Ndata)
        self.Uact = np.zeros((nu,Ndata))
        self.Xref = np.zeros((nx,Ndata))
        self.Uref = np.zeros((nu,Ndata))
        self.Xest = np.zeros((nx,Ndata))
        self.Xext = np.zeros((nx,Ndata))
        self.Adv  = np.zeros((nu,Ndata))
        self.Tsol = np.zeros((Ntsol,Ndata))
        self.k  = 0
        self.hz = hz

        # Process image function
        transform = A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])
        self.process_image = lambda x: transform(image=x)["image"]                  # Image Preprocessing
        
        # Save the output path and base name
        self.output_path = output_path
        self.output_type = "flight"
        self.output_base = course_name+"_"+pilot_name

    def record(self,
               img:np.ndarray,tcr:float,ucr:np.ndarray,
               xref:np.ndarray,uref:np.ndarray,xest:np.ndarray,xext:np.ndarray,
               adv:np.ndarray,tsol:np.ndarray):
        """"
        Record the flight data and images.
        
        Args:
        icr:    Current image
        tcr:    Current time
        ucr:    Current input
        xref:   Reference state
        uref:   Reference input
        xest:   Estimated state
        xext:   External state
        adv:    Advisor values
        tsol:   Solve times
        
        Returns:
        None

        """
        
        # Record the image data
        if self.k%self.n_img == 0:
            self.Imgs[int(self.k/self.n_img),:,:,:] = img

        # Record the trajectory data
        self.Tact[self.k]   = tcr
        self.Uact[:,self.k] = ucr
        self.Xref[:,self.k] = xref
        self.Uref[:,self.k] = uref
        self.Xest[:,self.k] = xest
        self.Xext[:,self.k] = xext
        self.Adv[:,self.k]  = adv
        self.Tsol[:,self.k] = tsol

        # Increment the counter
        self.k += 1
    
    def simulation_import(self,
                          Imgs:torch.Tensor,
                          Tro:np.ndarray,Xro:np.ndarray,Uro:np.ndarray,
                          tXUi:np.ndarray,Adv:np.ndarray,Tsol:np.ndarray):
        """
        Import the flight data from a trajectory rollout.
        
        Args:
            Imgs:   Image data
            Tro:    Actual time
            Xro:    Actual state
            Uro:    Actual input
            tXUi:   Reference state/input
            Adv:    Advisor values
            Tsol:   Solve times
        
        Returns:
            None

        """
        self.output_type = "sim"

        self.Imgs = Imgs
        self.Tact = Tro
        self.Uact = Uro
        self.Xref = tXUi[1:11,:]
        self.Uref = np.zeros_like(Uro)
        self.Xest = Xro
        self.Xext = Xro
        self.Adv = Adv
        self.Tsol = Tsol
        self.k = Tro.shape[0]-1

    def save(self):
        """
        Save the flight data and images to a .pt file. The data is trimmed to the
        correct size before saving.

        Args:
        None

        Returns:
        output_path:    Path to the saved file (for debugging purposes)

        """
        # Trim the flight data
        if self.k == 0:
            print("No data to save.")

            return None
        else:
            # Trim the trajectory data
            self.Imgs = self.Imgs[0:int(self.k/self.n_img),:,:,:]
            self.Tact = self.Tact[0:self.k]
            self.Uact = self.Uact[:,0:self.k]
            self.Xref = self.Xref[:,0:self.k]
            self.Uref = self.Uref[:,0:self.k]
            self.Xest = self.Xest[:,0:self.k]
            self.Xext = self.Xext[:,0:self.k]
            self.Adv  = self.Adv[:,0:self.k]
            self.Tsol = self.Tsol[:,0:self.k]

            # Create the output name
            output_name = self.output_type+"_"+self.output_base+"_"+datetime.now().strftime("%Y%m%d_%H%M%S")
            data_path = os.path.join(self.output_path,output_name+".pt")
            video_path = os.path.join(self.output_path,output_name+".mp4")

            # Create the .mp4
            if self.Imgs.shape[1] == 3:
                imgs_vd = np.transpose(self.Imgs, (0, 2, 3, 1))  # Reorder to (frames, height, width, channel)
            else:
                imgs_vd = self.Imgs
                
            imgs_vd = np.clip(imgs_vd, 0, 255).astype(np.uint8)
            
            # Write video using imageio
            with imageio.get_writer(video_path, format='FFMPEG', mode='I', fps=self.Nimps) as writer:
                for frame in imgs_vd:
                    writer.append_data(frame)
            
            # Save the data
            torch.save({
                'Imgs':self.process_images(self.Imgs),
                'Tact':self.Tact,
                'Uact':self.Uact,
                'Xref':self.Xref,
                'Uref':self.Uref,
                'Xest':self.Xest,
                'Xext':self.Xext,
                'Adv' :self.Adv,
                'Tsol':self.Tsol,
                'obj' :self.obj,
                'n_im':self.n_img
                },
                data_path)
        
            return data_path
    
    def process_images(self,images:np.ndarray):
        """
        Converts raw images to processed images using a specified process configuration.
        """
   
        output = torch.zeros((images.shape[0],3,224,224))
        for i in range(images.shape[0]):
            output[i,:,:,:] = self.process_image(images[i])

        return output