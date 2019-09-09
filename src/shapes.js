import * as tf from '@tensorflow/tfjs'
import * as R from 'ramda'
import * as C from './constants'
import * as I from './image'

const drawSquare = (ctx, boundingBox) => {
  ctx.fillStyle = 'pink'
  ctx.fillRect(...boundingBox)
  ctx.strokeStyle = 'gray'
  ctx.lineWidth = 2
  ctx.strokeRect(...boundingBox)
}

const drawTriangle = (ctx, boundingBox) => {
  const [x, y, w, h] = boundingBox
  const path = new Path2D()
  ctx.moveTo(x, y + h)
  ctx.lineTo(x + w / 2, y)
  ctx.lineTo(x + w, y + h)
  path.closePath()
  ctx.fillStyle = 'pink'
  ctx.strokeStyle = 'gray'
  ctx.lineWidth = 2
  ctx.fill()
}

const drawCircle = (ctx, boundingBox) => {
  const [x, y, w, h] = boundingBox
  const rx = w / 2
  const ry = h / 2
  ctx.ellipse(x + rx, y + ry, rx, ry, 0, 0, Math.PI * 2)
  ctx.fillStyle = 'pink'
  ctx.strokeStyle = 'gray'
  ctx.lineWidth = 2
  ctx.fill()
}

const SHAPE_DATA = [
  { shapeType: C.SHAPE_TYPE_SQUARE, drawingFunction: drawSquare },
  { shapeType: C.SHAPE_TYPE_TRIANGLE, drawingFunction: drawTriangle },
  { shapeType: C.SHAPE_TYPE_CIRCLE, drawingFunction: drawCircle }
]

const generateShape = () => {

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

  const shapeDataIndex = Math.floor(Math.random() * SHAPE_DATA.length)
  const { shapeType, drawingFunction } = SHAPE_DATA[shapeDataIndex]
  drawingFunction(ctx, boundingBox)

  const imageData = ctx.getImageData(0, 0, C.IMAGE_SIZE, C.IMAGE_SIZE)
  const imageTensor = I.normaliseImage(imageData)

  return {
    imageTensor,
    shapeType,
    boundingBox
  }
}

export const generateShapes = numShapes => {
  const shapes = R.times(generateShape, numShapes)
  const imageTensors = R.pluck('imageTensor', shapes)
  const shapeTypes = R.pluck('shapeType', shapes)
  const boundingBoxes = R.pluck('boundingBox', shapes)
  const outputs = R.zipWith(
    (shapeType, boundingBox) => [shapeType, ...boundingBox],
    shapeTypes,
    boundingBoxes)
  const xs = tf.stack(imageTensors)
  const ys = tf.tensor2d(outputs)
  return { shapes, xs, ys }
}
