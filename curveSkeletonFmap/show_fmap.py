# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def close_event():
  plt.close()
    
if __name__ == '__main__':
  fig = plt.figure()
  #timer = fig.canvas.new_timer(interval = 2000) 
  #timer.add_callback(close_event)
  
  file = open(sys.argv[1], "r")
  mat = []
  numeigs = int(sys.argv[2])
  for _ in range(numeigs):
    line = list(map(float, file.readline().split(" ")))
    if (not line): break
    mat.append(line)

  np_mat = np.array(mat)

  x = range(numeigs)
  y = range(numeigs)
  X, Y = np.meshgrid(x, y)

  plt.pcolormesh(X, Y, np_mat)
  plt.axis("image")
  plt.colorbar()

  fig, ax = plt.subplots()

  image = ax.pcolormesh(X, Y, np_mat)
  ax.axis("image")

  divider = make_axes_locatable(ax)
  ax_cb = divider.new_horizontal(size="2%", pad=0.05)
  fig.add_axes(ax_cb)
  plt.close()
  plt.colorbar(image, cax=ax_cb)
  #timer.start()
  plt.show()