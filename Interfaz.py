import threading
import queue
from tkinter import *
from tkinter import ttk

import cv2
import pydicom
import numpy as np
from PIL import Image, ImageTk
import os
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from scipy.ndimage import zoom, rotate

class GUI(Frame):

    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.parent = master
        self.grid(pady=10, padx=10)
        self.n_bins = 16
        self.flag_eqHist = False
        self.flag_seg = False
        self.sort_slices = True
        self.factor = 1
        self.createWidgets()

    def load_dcm(self, filename):
        # Lee ficheros Dicom.
        return pydicom.dcmread(f'P2 - DICOM/RM_Brain_3D-SPGR/{filename}')

    def load_imgs(self):
        # Se le indica el directorio de los ficheros Dicom y carga todas las imágenes.
        path = 'P2 - DICOM/RM_Brain_3D-SPGR/'
        self.img_names = sorted([f for f in os.listdir(path) if os.path.splitext(f)[-1] == '.dcm'])

        self.dcms = [self.load_dcm(f) for f in self.img_names]

        if self.sort_slices:
            # Reordenar según SliceLocation
            assert all(hasattr(slc, 'SliceLocation') for slc in self.dcms)
            slices = sorted(self.dcms, key=lambda s: s.SliceLocation)

        self.imgs = np.stack([slc.pixel_array for slc in slices])
        # Las imágenes se encuentran rotadas en un ángulo de 180º con respecto al paciente medio y atlas
        self.imgs = rotate(self.imgs, 180, reshape = False)
        self.pixel_len = [self.dcms[0].SliceThickness, self.dcms[0].PixelSpacing[0], self.dcms[0].PixelSpacing[1]]

    def dicom_header(self):
        # Se extrae la cabecera Dicom y se visualiza en una nueva ventana.
        win = Toplevel()
        win.wm_title('Cabecera fichero Dicom')

        scroll_bar = Scrollbar(win)
        scroll_bar.pack(side=RIGHT, fill=Y)

        mylist = Listbox(win, yscrollcommand=scroll_bar.set, width=0, height=0)
        [mylist.insert(END, line) for line in self.dcm_rest]
        mylist.pack(side=LEFT, fill=BOTH)

        scroll_bar.config(command=mylist.yview)

    def slice_selection(self, event):
        # Cambia el corte según el que haya seleccionado el usuario.
        # A su vez, restaura el valor de escalado de la imagen, pero
        # mantiene la ecualización y segmentación para apreciar el procesado
        # de imágenes en 3D.
        self.factor = 1

        self.text.set('Nivel de gris: -')

        idx = self.combo.current()
        self.dcm_rest = self.imgs[idx]

        self.img_h = self.dcm_rest.shape[0]
        self.img_w = self.dcm_rest.shape[1]

        self.imgM = self.convert_image()
        self.visualize(self.imgM)

    def equalize_histogram(self):
        # Según la selección del usuario, activa o desactiva
        # el flag que indica realizar la ecualización.
        if self.ec1.get() == 1:
            self.flag_eqHist = True
        else:
            self.flag_eqHist = False

        self.imgM = self.convert_image()
        self.visualize(self.imgM)

    @staticmethod
    def histogram(img_hist, queue, n_bins):
        # Calcula y dibuja el histograma.
        f = Figure(figsize=(5, 4), dpi=100)
        p = f.gca()
        p.set_title('Histograma')
        p.set_xlabel('Niveles de gris')
        p.set_ylabel('Frecuencia')

        if len(img_hist.shape) == 3:
            p.hist(img_hist[:, :, 0].ravel(), n_bins, [0, 256])
        else:
            p.hist(img_hist.ravel(), n_bins, [0, 256])

        queue.put(f)

    def number_bins_histogram(self, event):
        # Recoge el valor de bins seleccionado por el usuario
        # y llama a la función para graficar el histograma.
        self.n_bins = int(self.combo2.get())
        self.imgM = self.convert_image()
        self.visualize(self.imgM)

    def motion(self, imgMov, x1, y1):
        # Recoge el valor de intensidad del píxel dónde se encuentra
        # el cursor.
        pixel_value = '-'
        if (y1 < imgMov.shape[0]) and (x1 < imgMov.shape[1]):
            pixel_value = imgMov[y1, x1]
        else:
            pixel_value = '-'

        self.text.set('Nivel de gris: ' + str(pixel_value))

    def move_start(self, event):
        # Llama a la función de segmentación en el caso
        # de que el usuario haya pulsado el botón de
        # seleccionar píxel.
        if self.cb1.get() > 0:
            self.get_value_segmentation(event)
        else:
            self.img_canvas.scan_mark(event.x, event.y)

    def move_move(self, event):
        # Permite desplazarse por la imagen realizando un "drag"
        # (pulsar y desplazar).
        self.img_canvas.scan_dragto(event.x, event.y, gain=1)

    def zoom_IN_OUT(self, in_out):
        # Al realizar "scroll" en el ratón, esta función recoge
        # qué tipo de "scroll" se ha realizado y acumula esa cantidad
        # para calcular un "resize" de la imagen.
        if in_out:
            if self.factor <= 2.5:
                self.factor += 0.05
            self.img_canvas.delete(self.id_img)
            self.imgM = self.convert_image()
            self.visualize(self.imgM)
        else:
            if self.factor > 1:
                self.factor -= 0.05
            if self.factor == 1:
                self.img_canvas.delete(self.id_img)
            self.imgM = self.convert_image()
            self.visualize(self.imgM)

    def get_value_segmentation(self, event):
        # Recoge el valor del píxel seleccionado que se utilizará
        # para segmentar.
        x = int(self.img_canvas.canvasx(event.x))
        y = int(self.img_canvas.canvasy(event.y))

        self.flag_seg = False
        img_tmp = self.convert_image()
        self.valor = img_tmp[y, x]
        self.flag_seg = True

        self.imgM = self.convert_image()
        self.visualize(self.imgM)

    def iso_contour(self):
        # Si deselecciónan el botón de seleccionar píxel
        # desactiva la segmentación.
        if self.cb1.get() == 0:
            self.flag_seg = False
            self.imgM = self.convert_image()
            self.visualize(self.imgM)

    def convert_image(self):
        # Se realizan todas las operaciones sobre la imagen de trabajo según
        # el estado de los flags.
        array = self.dcm_rest
        img_zoom = cv2.resize(array, (int(self.img_h * self.factor), int(self.img_w * self.factor)),
                              interpolation=cv2.INTER_CUBIC)
        if self.flag_eqHist:
            img_RGB = ((img_zoom / 5100) * 256).astype(np.uint8)
            img_RGB = cv2.equalizeHist(img_RGB)
        else:
            img_RGB = ((img_zoom / 5100) * 256).astype(np.uint8)

        if self.flag_seg:
            img_th = img_RGB >= self.valor
            img_RGB = cv2.cvtColor(img_RGB, cv2.COLOR_GRAY2RGB)

            img_mask = np.zeros_like(img_RGB, dtype=np.uint8)
            img_mask[img_th, 0] = 255

            img_th_neg = np.logical_not(img_th)
            img_mask[img_th_neg] = img_RGB[img_th_neg]

            img_RGB = img_mask

        return img_RGB

    def output(self):
        # Para mejorar la experiencia de usuario, se procede a generar un "thread" o hilo
        # para el cálculo en paralelo del histograma, ya que se trata de un cálculo
        # que requiere de tiempo sobretodo al engrandar la imagen con zoom.
        try:
            message = self.queue.get(0)
            self.canvas = FigureCanvasTkAgg(message, self)
            self.canvas.get_tk_widget().grid(row=5, column=0, columnspan=8, sticky=NSEW)
            self.canvas.draw()
        except queue.Empty:
            self.master.after(50, self.output)

    def visualize(self, img_out):
        # Se grafica la imagen que en ese momento se está trabajando.
        self.queue = queue.Queue()
        proceso = threading.Thread(target = self.histogram, args = [img_out, self.queue, self.n_bins]).start()
        self.master.after(50, self.output)

        # Al colocar la imagen en un "Canvas" es necesario convertir la imagen a un
        # formato específico.
        img_pil = Image.fromarray(img_out)
        self.imgTk = ImageTk.PhotoImage(image=img_pil)

        self.img_canvas = Canvas(self, width=self.dcm_rest.shape[1],
                                 height=self.dcm_rest.shape[0])

        # Los "scrollbar" son útiles cuando se está segmentando y se desea desplazar
        # a lo largo de la imagen que está agrandada.
        self.xsb = Scrollbar(self, orient='horizontal', command=self.img_canvas.xview)
        self.ysb = Scrollbar(self, orient='vertical', command=self.img_canvas.yview)
        self.img_canvas.configure(yscrollcommand=self.ysb.set, xscrollcommand=self.xsb.set)
        self.img_canvas.configure(scrollregion=(0, 0, 1000, 1000))

        self.xsb.grid(row=9, column=9, sticky='ew')
        self.ysb.grid(row=0, column=10, rowspan=8, sticky='ns')

        self.id_img = self.img_canvas.create_image(0, 0, image=self.imgTk, anchor=NW)
        self.img_canvas.grid(row=0, column=9, rowspan=8, sticky=NSEW)
        self.img_canvas.configure(scrollregion=self.img_canvas.bbox('all'))

        # Se recoge las coordenadas imagen para calcular su posterior
        # intensidad de píxel.
        def preMotion(event):
            self.x1 = int(self.img_canvas.canvasx(event.x))
            self.y1 = int(self.img_canvas.canvasy(event.y))
            return self.motion(img_out, self.x1, self.y1)

        self.img_canvas.bind('<Motion>', preMotion)

        self.img_canvas.bind('<ButtonPress-1>', self.move_start)
        self.img_canvas.bind('<B1-Motion>', self.move_move)

        # Se recoge que tipo de "scroll" se ha realizado.
        def zoomer(event):
            if (event.delta > 0):
                in_out = True
                return self.zoom_IN_OUT(in_out)
            else:
                in_out = False
                return self.zoom_IN_OUT(in_out)

        self.img_canvas.bind('<MouseWheel>', zoomer)

    def createWidgets(self):

        # Cargar imágenes
        self.load_imgs()
        self.dcm_rest = self.imgs[0]
        self.img_h = self.dcm_rest.shape[0]
        self.img_w = self.dcm_rest.shape[1]
        self.imgM = self.convert_image()
        self.visualize(self.imgM)

        # Etiqueta seleccionar corte
        self.l0 = Label(self, text='Seleccionar corte')
        self.l0.grid(row=0, column=0, sticky=NSEW)

        # Menú para seleccionar el corte
        self.box_value = StringVar()
        self.box_value.set(self.img_names[0])
        self.combo = ttk.Combobox(self, textvariable=self.box_value, state='readonly')
        self.combo['values'] = self.img_names
        self.combo.bind('<<ComboboxSelected>>', self.slice_selection)
        self.combo.grid(row=0, column=1, sticky=NSEW)

        # Botón para obtener la cabecera Dicom
        self.dcm_header = Button(self, text='Cabecera Dicom', command=self.dicom_header)
        self.dcm_header.grid(row=1, column=0, sticky=NSEW)

        # Etiqueta ecualizar imagen
        self.l1 = Label(self, text='Ecualizar imagen')
        self.l1.grid(row=2, column=0, sticky=NSEW)

        # Botón para activar la ecualización de la imagen
        self.ec1 = IntVar()
        self.ecualizar = Checkbutton(self, variable=self.ec1, command=self.equalize_histogram)
        self.ecualizar.grid(row=2, column=1, sticky=NSEW)

        # Etiqueta seleccionar píxel
        self.l2 = Label(self, text='Seleccionar píxel')
        self.l2.grid(row=3, column=0, sticky=NSEW)

        # Botón para activar la segmentación
        self.cb1 = IntVar()
        self.segmentacion = Checkbutton(self, variable=self.cb1, command = self.iso_contour)
        self.segmentacion.grid(row=3, column=1, sticky=NSEW)

        # Etiqueta número de bins del histograma
        self.l4 = Label(self, text='Número de bins')
        self.l4.grid(row=4, column=0, sticky=NSEW)

        # Menú para seleccionar el número de bins del histograma
        self.box_value2 = StringVar()
        self.box_value2.set('16')
        self.combo2 = ttk.Combobox(self, textvariable=self.box_value2, state='readonly')
        out = [2 ** j for j in range(4, 9)]
        self.combo2['values'] = out
        self.combo2.bind('<<ComboboxSelected>>', self.number_bins_histogram)
        self.combo2.grid(row=4, column=1, sticky=NSEW)

        # Valor etiqueta nivel de gris
        self.text = StringVar()
        self.text.set('Nivel de gris: -')

        # Etiqueta nivel de gris
        self.l3 = Label(self, textvariable=self.text)
        self.l3.grid(row=10, column=9, sticky=NSEW)


Medical_gui = Tk()
Medical_gui.title('Interfaz de imagenes')
root = GUI(Medical_gui).grid()
Medical_gui.mainloop()