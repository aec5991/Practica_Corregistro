import pydicom
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import zoom, rotate
import os
import cv2

import math
from scipy.optimize import least_squares

# Se carga el fichero atlas
atlas_dcm = pydicom.dcmread('AAL3_1mm.dcm')
print('Dimensiones atlas:', atlas_dcm.pixel_array.shape)

# Se carga el fichero paciente normalizado o 'paciente medio'
paciente_medio_sym = pydicom.dcmread('icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm')
print('Dimensiones paciente medio:', paciente_medio_sym.pixel_array.shape)

#paciente_medio_alt = pydicom.dcmread('icbm_avg_152_t1_tal_nlin_symmetric_VI_alternative.dcm')
#print('Tamaño paciente medio alt:', paciente_medio_alt.pixel_array.shape)
print()

# Se convierte el atlas y paciente medio a formato imagen
atlas = atlas_dcm.pixel_array
p_medio = paciente_medio_sym.pixel_array

# Como el paciente medio y el atlas no tienen las mismas dimensiones se deben interpolar y reescalar.
# Lo lógico es interpolar y reescalar el atlas para que encaje con las dimensiones del paciente medio.
f1_atlas = p_medio.shape[0]/atlas.shape[0]
f2_atlas = p_medio.shape[1]/atlas.shape[1]
f3_atlas = p_medio.shape[2]/atlas.shape[2]

medial = p_medio.shape[0]//2

atlas_resize = zoom(atlas, (f1_atlas, f2_atlas, f3_atlas))
print('Dimensiones atlas reescalado:', atlas_resize.shape, '¿==?',
      p_medio.shape, ':Dimensiones paciente medio')

# Se superpone el atlas con el paciente medio para saber qué regiones del cerebro se corresponden
fusion_alpha = cv2.addWeighted(p_medio[medial], 0.7, atlas_resize[medial], 0.3, 0.0)

plt.figure()
plt.subplot(131)
plt.title('Slice Atlas')
plt.imshow(atlas_resize[medial])

plt.subplot(132)
plt.title('Slice Paciente Medio')
plt.imshow(p_medio[medial])

plt.subplot(133)
plt.title('Fusion Alpha')
plt.imshow(fusion_alpha)
plt.show()

# Se carga el fichero del paciente
files = os.listdir('P2 - DICOM/RM_Brain_3D-SPGR/')
files.sort(reverse=True)
slices = [pydicom.dcmread(f'P2 - DICOM/RM_Brain_3D-SPGR//{file}')
          for file in files]

# Se ordena según SliceLocation
assert all(hasattr(slc, 'SliceLocation') for slc in slices)
slices = sorted(slices, key=lambda s: s.SliceLocation)

# Se convierten los cortes a formato imagen
img = np.stack([slc.pixel_array for slc in slices])

print('Dimensiones MRI paciente:', img.shape)

medial_MRI = img.shape[0]//2

plt.figure()
plt.title('Slice Paciente')
plt.imshow(img[medial_MRI])
plt.show()

# Se debe rotar 180º
img = rotate(img, 180, reshape = False)

# Se extraen las características de las imágenes: mm entre cortes y tamaño píxeles
pixel_len = [slices[0].SliceThickness, slices[0].PixelSpacing[0], slices[0].PixelSpacing[1]]
print('Caracteristicas escalado:', pixel_len)
print()

plt.figure()
plt.subplot(131)
plt.title('Slice Atlas')
plt.imshow(atlas_resize[medial])

plt.subplot(132)
plt.title('Slice Paciente Medio')
plt.imshow(p_medio[medial])

plt.subplot(133)
plt.title('Slice Paciente Rotado')
plt.imshow(img[medial_MRI])
plt.show()

# De forma análoga, hay que corresponder las dimensiones del paciente con el paciente medio:
# primero hay que convertirlo a escala 1 y después ajustar los cortes a la dimensión del paciente medio.
f1_img1 = pixel_len[0]
f2_img1 = pixel_len[1]
f3_img1 = pixel_len[2]

# Al pasar de escala 2 a escala 1, se obtendrán el doble de cortes y la resolución será más grande.
img_resize1 = zoom(img, (f1_img1, f2_img1, f3_img1))
print('Dimensiones imagenes paciente reescalado:', img_resize1.shape, '¿==?',
      p_medio.shape, ':Dimensiones paciente medio')

# Para reajustar el tamaño del cerebro se hace un "cropping" centrado
crop_rows = (img_resize1.shape[1] - p_medio.shape[1])//2
crop_cols = (img_resize1.shape[2] - p_medio.shape[2])//2

img_crop = img_resize1[:, crop_rows:img_resize1.shape[1] - crop_rows, crop_cols:img_resize1.shape[2] - crop_cols]
print('Dimensiones imagenes paciente crop:', img_crop.shape, '¿==?',
      p_medio.shape, ':Dimensiones paciente medio')

# Debido a las divisiones enteras, es posible que haya quedado alguna columna o fila desajustada en una
# unidad. Por eso se realiza un pequeño ajuste final.
f1_img_crop = p_medio.shape[0]/img_crop.shape[0]
f2_img_crop = p_medio.shape[1]/img_crop.shape[1]
f3_img_crop = p_medio.shape[2]/img_crop.shape[2]

img_final = zoom(img_crop, (f1_img_crop, f2_img_crop, f3_img_crop))
print('Dimensiones imagenes paciente (final):', img_final.shape, '¿==?',
      p_medio.shape, ':Dimensiones paciente medio')

plt.figure()
plt.subplot(131)
plt.title('Slice Atlas')
plt.imshow(atlas_resize[medial])

plt.subplot(132)
plt.title('Slice Paciente Medio')
plt.imshow(p_medio[medial])

plt.subplot(133)
plt.title('Slice Paciente')
plt.imshow(img_final[medial])
plt.show()


# CORREGISTRO incompleto
def traslacion(punto, vector_traslacion):
    x, y, z = punto
    t_1, t_2, t_3 = vector_traslacion
    punto_transformado = (x+t_1, y+t_2, z+t_3)
    return punto_transformado

def rotacion_axial(punto, angulo_en_radianes, eje_traslacion):
    x, y, z = punto
    v_1, v_2, v_3 = eje_traslacion
    #   Vamos a normalizarlo para evitar introducir restricciones en el optimizador
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v_1, v_2, v_3]]))
    v_1, v_2, v_3 = v_1 / v_norm, v_2 / v_norm, v_3 / v_norm
    #   Calcula cuaternión del punto
    p = (0, x, y, z)
    #   Calcula cuaternión de la rotación
    cos, sin = math.cos(angulo_en_radianes / 2), math.sin(angulo_en_radianes / 2)
    q = (cos, sin * v_1, sin * v_2, sin * v_3)
    #   Calcula el conjugado
    q_conjugado = (cos, -sin * v_1, -sin * v_2, -sin * v_3)
    #   Calcula el cuaternión correspondiente al punto rotado
    p_prima = multiplicar_quaterniones(q, multiplicar_quaterniones(p, q_conjugado))
    # Devuelve el punto rotado
    punto_transformado = p_prima[1], p_prima[2], p_prima[3]
    return punto_transformado

def transformacion_rigida_3D(punto, parametros):
    x, y, z = punto
    t_11, t_12, t_13, alpha_in_rad, v_1, v_2, v_3, t_21, t_22, t_23 = parametros
    #   Aplicar una primera traslación
    x, y, z = traslacion(punto=(x, y, z), vector_traslacion=(t_11, t_12, t_13))
    #   Aplicar una rotación axial traslación
    x, y, z = rotacion_axial(punto=(x, y, z), angulo_en_radianes=alpha_in_rad, eje_traslacion=(v_1, v_2, v_3))
    #   Aplicar una segunda traslación
    x, y, z = traslacion(punto=(x, y, z), vector_traslacion=(t_21, t_22, t_23))
    punto_transformado = (x, y, z)
    return punto_transformado

def multiplicar_quaterniones(q1, q2):
    """Multiplica cuaterniones expresados como (1, i, j, k)."""
    return (
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    )

def cuaternion_conjugado(q):
    """Conjuga un cuaternión expresado como (1, i, j, k)."""
    return (
        q[0], -q[1], -q[2], -q[3]
    )

def residuos_cuadraticos(lista_puntos_ref, lista_puntos_inp):
    """Devuelve un array con los residuos cuadráticos del ajuste."""
    residuos = []
    for p1, p2 in zip(lista_puntos_ref, lista_puntos_inp):
        p1 = np.asarray(p1, dtype='float')
        p2 = np.asarray(p2, dtype='float')
        residuos.append(np.sqrt(np.sum(np.power(p1-p2, 2))))
    residuos_cuadraticos = np.power(residuos, 2)
    return residuos_cuadraticos

parametros_iniciales = [0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 0]

for i in range(img_final.shape[0]):
    for j in range(img_final.shape[1]):
        for k in range(img_final.shape[2]):
            inp_transf = transformacion_rigida_3D((i, j, k), parametros_iniciales)
            a = int(inp_transf[0])
            b = int(inp_transf[1])
            c = int(inp_transf[2])
            values = p_medio[a][b][c]

# Está incompleto
# def funcion_a_minimizar(parametros):
#     # Adaptar para una imagen
#     landmarks_inp_transf = [transformacion_rigida_3D(landmark, parametros) for landmark in landmarks_inp]
#     return residuos_cuadraticos(landmarks_ref, landmarks_inp_transf)
#
# resultado = least_squares(funcion_a_minimizar,
#                           x0=parametros_iniciales,
#                           verbose=1)
# x_opt = resultado.x
# print(f'''
# Los mejores parámetros son:
#     1) Traslación respecto al vector ({x_opt[0]}, {x_opt[1]}, {x_opt[2]}).
#     2) Rotación axial de {math.degrees(x_opt[3])} grados, eje ({x_opt[4]}, {x_opt[5]}, {x_opt[6]}).
#     3) Traslación respecto al vector ({x_opt[7]}, {x_opt[8]}, {x_opt[9]}).
# ''')