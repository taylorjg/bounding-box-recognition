import * as tf from '@tensorflow/tfjs'
import * as R from 'ramda'
import * as C from './constants'

export const convertToGreyscale = imageData => {
  const width = imageData.width
  const height = imageData.height
  const numPixels = width * height
  const data = imageData.data
  const array = new Uint8ClampedArray(data.length)
  const bases = R.range(0, numPixels).map(index => index * 4)
  for (const base of bases) {
    const colourValues = data.slice(base, base + 4)
    const [r, g, b, a] = colourValues
    // https://imagemagick.org/script/command-line-options.php#colorspace
    // Gray = 0.212656*R+0.715158*G+0.072186*B
    const greyValue = 0.212656 * r + 0.715158 * g + 0.072186 * b
    array[base] = greyValue
    array[base + 1] = greyValue
    array[base + 2] = greyValue
    array[base + 3] = a
  }
  return new ImageData(array, width, height)
}

export const normaliseImage = imageData => {
  const imageDataGreyscale = convertToGreyscale(imageData)
  return tf.browser.fromPixels(imageDataGreyscale, C.IMAGE_CHANNELS)
}
