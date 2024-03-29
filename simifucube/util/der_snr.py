# This file is part of Simifucube.
#
# Copyright (C) 2007-2019 Felix Stoehr, ST-ECF (fstoehr@eso.org)
# Copyright (C) 2019 Michele Mastropietro (michele.mastropietro@gmail.com)
#
# Simifucube is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Simifucube is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Simifucube.  If not, see <https://www.gnu.org/licenses/>.


# Adapted from www.stecf.org/software/ASTROsoft/DER_SNR/

def DER_SNR(flux):

# =====================================================================================
   """
   DESCRIPTION This function computes the signal to noise ratio DER_SNR following the
               definition set forth by the Spectral Container Working Group of ST-ECF,
               MAST and CADC.

               signal = median(flux)
               noise  = 1.482602 / sqrt(6) median(abs(2 flux_i - flux_i-2 - flux_i+2))
               snr    = signal / noise
               values with padded zeros are skipped

   USAGE       snr = DER_SNR(flux)
   PARAMETERS  none
   INPUT       flux (the computation is unit independent)
   OUTPUT      the estimated signal-to-noise ratio [dimensionless]
   USES        numpy
   NOTES       The DER_SNR algorithm is an unbiased estimator describing the spectrum
               as a whole as long as
               * the noise is uncorrelated in wavelength bins spaced two pixels apart
               * the noise is Normal distributed
               * for large wavelength regions, the signal over the scale of 5 or
                 more pixels can be approximated by a straight line

               For most spectra, these conditions are met.

   REFERENCES  * ST-ECF Newsletter, Issue #42:
               www.spacetelescope.org/about/further_information/newsletters/html/newsletter_42.html
               https://www.spacetelescope.org/static/archives/stecfnewsletters/pdf/hst_stecf_0042.pdf
               * Software:
               www.stecf.org/software/ASTROsoft/DER_SNR/
   AUTHOR      Felix Stoehr, ST-ECF
               24.05.2007, fst, initial import
               01.01.2007, fst, added more help text
               28.04.2010, fst, return value is a float now instead of a numpy.float64
   """
   from numpy import array, where, nanmedian, abs

   flux = array(flux)

   # Values that are exactly zero (padded) are skipped
   flux = array(flux[where(flux != 0.0)])
   n    = len(flux)

   # For spectra shorter than this, no value can be returned
   if (n>4):
      signal = nanmedian(flux)

      noise  = 0.6052697 * nanmedian(abs(2.0 * flux[2:n-2] - flux[0:n-4] - flux[4:n]))

      return noise

   else:

      return 0.0

# end DER_SNR -------------------------------------------------------------------------
