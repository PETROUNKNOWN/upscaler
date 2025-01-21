import warnings
import os
import customtkinter as ctk
import torch
import numpy as np
import time
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
warnings.filterwarnings("ignore")

class Upscaler:
    def __init__(self, root):
        self.root=root
        self.root.title("pyUpscaler")
        self.root.geometry("+-5+0")
        self.root.resizable(0,0)
        self.root.columnconfigure(0,weight=1)
        self.root.rowconfigure(0,weight=1)

        main_frame=ctk.CTkFrame(root,fg_color="#101010")
        main_frame.grid(row=0,column=0,sticky="nsew")
        main_frame.columnconfigure(0,weight=1)
        main_frame.columnconfigure(1,weight=1)

        self.modelFileEntry=ctk.CTkEntry(main_frame,width=500,height=40,placeholder_text="Model File Name",fg_color="#101010",border_color="#ff0000",border_width=1)
        self.modelFileEntry.grid(row=0,column=0,columnspan=2,sticky="nsew",pady=(10,10),padx=(10,10))

        self.imageFileEntry=ctk.CTkEntry(main_frame,width=500,height=40,placeholder_text="Image File Name",fg_color="#101010",border_color="#ff0000",border_width=1)
        self.imageFileEntry.grid(row=1,column=0,columnspan=2,sticky="nsew",pady=(0,10),padx=(10,10))

        upscaleLabel=ctk.CTkLabel(main_frame,text="Upscale Value")
        upscaleLabel.grid(row=2,column=0,sticky="nsw",pady=(0,10),padx=(10,10))
        upscaleframe=ctk.CTkFrame(main_frame,fg_color="#101010",border_color="#ff0000",border_width=1)
        upscaleframe.grid(row=2,column=1,sticky="nsew",pady=(0,10),padx=(10,10))
        upscaleframe.columnconfigure(0,weight=1)
        upscaleframe.rowconfigure(0,weight=1)
        self.upscaleFactorEntry = ctk.CTkOptionMenu(upscaleframe,width=400,values=["1","2","3","4","5","6","7","8","9","10","12","14","16","18","20"],button_color="#101010",fg_color="#101010",button_hover_color="#990000")
        self.upscaleFactorEntry.grid(row=0,column=0,sticky="nsew",pady=3,padx=3)

        self.button=ctk.CTkButton(main_frame,width=130,height=40,text="Run Model",command=self.validateEntry,fg_color="#101010",border_color="#ff0000",border_width=1,hover_color="#990000")
        self.button.grid(row=3,column=0,columnspan=2,sticky="nsew",pady=(0,10),padx=(10,10))

        self.console=ctk.CTkTextbox(main_frame,height=300,wrap="word",fg_color="#101010",border_color="#ff0000",border_width=1)
        self.console.grid(row=4,column=0,columnspan=2,sticky="nsew",pady=(0,10),padx=(10,10))

    def log_to_console(self,message):
        self.console.insert("end",f"{message}\n")
        self.console.see("end")

    def validateEntry(self):
        modelFileEntry=str(self.modelFileEntry.get())
        imageFileEntry=str(self.imageFileEntry.get())
        upscaleFactorEntry=int(self.upscaleFactorEntry.get())
        # print(modelFileEntry)
        # print(imageFileEntry)
        # print(upscaleFactorEntry)
        if not modelFileEntry or not imageFileEntry or not upscaleFactorEntry:
            self.log_to_console("Error: Please fill in all entry fields.")
            return
        
        # time.sleep(1)
        # self.log_to_console("Warning: T- 0s")
        try:
            self.log_to_console(f"Note: Model File Name: <{modelFileEntry}>")
            self.log_to_console(f"Note: Image File Name: <{imageFileEntry}>")
            self.log_to_console(f"Note: Upscale Factor: <{upscaleFactorEntry}>")
            time.sleep(1)
            self.runUpscale()
            
        except Exception as e:
            self.log_to_console(f"Error: An unexpected error occured: {e}")

    def runUpscale(self):
        self.log_to_console("Note: Model has started running.")
        model_path = os.path.join(os.path.dirname(__file__), str(self.modelFileEntry.get()))
        state_dict=torch.load(model_path, map_location=torch.device("cpu"))["params_ema"]
        model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=8)
        model.load_state_dict(state_dict, strict=True)
        upsampler=RealESRGANer(
            scale=3,
            model_path=model_path,
            model=model,
            tile=0,
            pre_pad=0,
            half=False
        )
        myImage=os.path.join(os.path.dirname(__file__), str(self.imageFileEntry.get()))
        img = Image.open(myImage).convert("RGB")
        img = np.array(img)

        output, _ = upsampler.enhance(img, outscale=int(self.upscaleFactorEntry.get()))

        myOutput=os.path.join(os.path.dirname(__file__), 'output.png')
        output_img = Image.fromarray(output)
        output_img.save(myOutput)
        self.log_to_console("Note: Model has completed running.")



if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    app_root=ctk.CTk()
    app=Upscaler(app_root)
    app_root.mainloop()


# works on small images fairly fast,,
# be ready to NUKE! your system if you use a big picture, also helps to have a powerful GPU,
# The script keeps working in an environment not where it is located.
# Input gets worked on then the output is thrown into the oblivion
# This temporary solution seems to fix it: the <os.path.join> thing