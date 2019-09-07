import * as tf from '@tensorflow/tfjs'
import * as R from 'ramda'
import * as C from './constants'
import * as I from './image'

const generateSquare = cb => index => {
  const upperCoordLimit = C.IMAGE_SIZE - 2 * C.SHAPE_MARGIN - C.SHAPE_MIN_SIZE
  const x = Math.random() * upperCoordLimit + C.SHAPE_MARGIN
  const y = Math.random() * upperCoordLimit + C.SHAPE_MARGIN
  const maxCoord = Math.max(x, y)
  const upperSizeLimit = C.IMAGE_SIZE - maxCoord - C.SHAPE_MARGIN - C.SHAPE_MIN_SIZE
  const size = Math.random() * upperSizeLimit + C.SHAPE_MIN_SIZE;
  const boundingBox = [x, y, size, size]

  const canvas = document.createElement('canvas')
  canvas.width = C.IMAGE_SIZE
  canvas.height = C.IMAGE_SIZE
  const ctx = canvas.getContext('2d')

  ctx.fillStyle = 'white'
  ctx.fillRect(0, 0, C.IMAGE_SIZE, C.IMAGE_SIZE)

  ctx.fillStyle = 'pink'
  ctx.fillRect(...boundingBox)

  ctx.strokeStyle = 'gray'
  ctx.lineWidth = 2
  ctx.strokeRect(...boundingBox)

  const imageData = ctx.getImageData(0, 0, C.IMAGE_SIZE, C.IMAGE_SIZE)
  const imageTensor = I.normaliseImage(imageData)

  const shape = {
    imageTensor,
    boundingBox
  }

  cb && cb(shape, index)

  return shape
}

export const generateShapes = (numShapes, cb) => {
  const shapes = R.times(generateSquare(cb), numShapes)
  const imageTensors = R.pluck('imageTensor', shapes)
  const boundingBoxes = R.pluck('boundingBox', shapes)
  const xs = tf.stack(imageTensors)
  const ys = tf.tensor2d(boundingBoxes)
  return { xs, ys }
}
