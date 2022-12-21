import numpy as np
import matplotlib.pyplot as plt
from _imfits import Imfits
import drawmaps as dm



def main():
	f = 'testfits/l1489.c18o.contsub.gain01.rbp05.mlt100.cf15.pbcor.mom0.trimed.fits'

	im = Imfits(f, frame='icrs')

	def _foward(x):
		#return np.sqrt(x)
		return np.arcsinh(x)

	def _inverse(x):
		#return x**2
		return np.arcsinh(x)

	color_norm = (_foward, _inverse)
	ax = dm.intensitymap(im, outname='test', 
		color_norm=color_norm, imscale=[-1,1,-1,1],
		xticks=np.arange(-1,1.1,0.5), yticks=np.arange(-1,1.1,0.5),
		cbaroptions=['top', '3%', '0%', 'Jy/beam'])#, vmin=-0.01, vmax=0.01)
	#ax.plot([0.0,0.5],[-0.7, -0.7],color='k',lw=3)
	#plt.savefig('test.pdf')
	plt.show()


if __name__ == '__main__':
	main()
