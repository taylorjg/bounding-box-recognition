import * as tf from '@tensorflow/tfjs'

export const drawBoundingBox = (canvas, boundingBox, colour) => {
  const ctx = canvas.getContext('2d')
  ctx.strokeStyle = colour
  ctx.lineWidth = 1
  ctx.strokeRect(...boundingBox)
}

export const drawImageTensor = async (parentElement, imageTensor) => {
  const canvas = document.createElement('canvas')
  canvas.setAttribute('class', 'image-canvas')
  await tf.browser.toPixels(imageTensor, canvas)
  parentElement.appendChild(canvas)
  return canvas
}
