
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from i18n import I18N

class ConversionTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=20)
        self.app = app
        self.build_ui()
        
    def build_ui(self):
        from main import REPO_ROOT
        
        ttk.Label(self, text=I18N[self.app.lang]["source_path"]).pack(anchor="w")
        src_frame = ttk.Frame(self)
        src_frame.pack(fill="x", pady=5)
        self.conv_src_entry = ttk.Entry(src_frame)
        self.conv_src_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(src_frame, text=I18N[self.app.lang]["browse"], command=self.browse_conv_src).pack(side="right", padx=5)
        
        ttk.Label(self, text=I18N[self.app.lang]["char_name"]).pack(anchor="w")
        self.conv_name_entry = ttk.Entry(self)
        self.conv_name_entry.pack(fill="x", pady=5)
        
        ttk.Label(self, text=I18N[self.app.lang]["output_root"]).pack(anchor="w")
        out_frame = ttk.Frame(self)
        out_frame.pack(fill="x", pady=5)
        self.conv_out_entry = ttk.Entry(out_frame)
        self.conv_out_entry.pack(side="left", fill="x", expand=True)
        self.conv_out_entry.insert(0, str(REPO_ROOT / "CharacterData" / "character_model" / "v2"))
        ttk.Button(out_frame, text=I18N[self.app.lang]["browse"], command=self.browse_conv_out).pack(side="right", padx=5)
        
        self.btn_convert = ttk.Button(self, text=I18N[self.app.lang]["convert"], style="Primary.TButton", command=self.convert_model)
        self.btn_convert.pack(pady=20)

    def browse_conv_src(self):
        path = filedialog.askdirectory()
        if path:
            self.conv_src_entry.delete(0, tk.END)
            self.conv_src_entry.insert(0, path)

    def browse_conv_out(self):
        path = filedialog.askdirectory()
        if path:
            self.conv_out_entry.delete(0, tk.END)
            self.conv_out_entry.insert(0, path)

    def convert_model(self):
        from lunavox_tts import convert_to_onnx
        from lunavox_tts.Converter.v2.Converter import find_ckpt_and_pth
        
        src = self.conv_src_entry.get()
        name = self.conv_name_entry.get()
        out_root = self.conv_out_entry.get()
        
        if not (src and name and out_root):
            messagebox.showwarning("Warning", "All fields are required")
            return
            
        def task():
            try:
                ckpt, pth = find_ckpt_and_pth(src)
                if not (ckpt and pth):
                    raise FileNotFoundError("Could not find .ckpt and .pth in source directory")
                
                dest_dir = os.path.join(out_root, name)
                self.app.set_status(f"Converting {name}...")
                convert_to_onnx(ckpt, pth, dest_dir)
                self.app.set_status(f"Conversion complete: {name}")
                messagebox.showinfo("Success", f"Model converted successfully to {dest_dir}")
                # Trigger refresh in inference tab if possible
                if hasattr(self.app, 'inf_tab'):
                    self.app.inf_tab.update_character_list()
            except Exception as e:
                self.app.set_status(f"Conversion failed: {e}")
                messagebox.showerror("Error", str(e))
        
        threading.Thread(target=task).start()
