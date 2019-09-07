import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import log from 'loglevel'
import * as C from './constants'
import * as U from './utils'
import * as DC from './drawCanvas'
import { generateShapes } from './shapes'

let model = undefined
let trained = false
let visor = undefined

const getVisor = () => {
  if (!visor) {
    visor = tfvis.visor()
  }
  visor.open()
  return visor
}

const showVisorBtn = document.getElementById('show-visor-btn')
const trainModelBtn = document.getElementById('train-model-btn')
const saveModelBtn = document.getElementById('save-model-btn')
const loadModelBtn = document.getElementById('load-model-btn')
const clearTrainingDataBtn = document.getElementById('clear-training-data-btn')
const makePredictionsBtn = document.getElementById('make-predictions-btn')
const clearPredictionsBtn = document.getElementById('clear-predictions-btn')
const trainingDataElement = document.getElementById('training-data')
const predictionsElement = document.getElementById('predictions')

const onShowVisor = () => {
  visor && visor.open()
}

const createModel = () => {
  const model = tf.sequential()
  model.add(tf.layers.conv2d({
    inputShape: [C.IMAGE_SIZE, C.IMAGE_SIZE, C.IMAGE_CHANNELS],
    kernelSize: 3,
    filters: 16,
    activation: 'relu'
  }))
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))
  model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }))
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))
  model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }))
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))
  model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }))
  model.add(tf.layers.flatten())
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }))
  model.add(tf.layers.dense({ units: 4 }))
  return model
}

// TODO: add try/catch/finally
// TODO: disposables
const onTrainModel = async () => {
  trainModelBtn.disabled = true
  U.deleteChildren(trainingDataElement)
  const cb = async (shape, index) => {
    if (index >= 10) return
    const canvas = await DC.drawImageTensor(trainingDataElement, shape.imageTensor)
    DC.drawBoundingBox(canvas, shape.boundingBox, 'blue')
  }
  const { xs, ys } = generateShapes(100, cb)

  model = createModel()

  model.compile({
    loss: 'meanAbsoluteError',
    optimizer: tf.train.rmsprop(5e-3)
  })

  const trainingSurface = getVisor().surface({
    tab: 'Bounding Box Recognition',
    name: 'Model Training'
  })

  const customCallback = tfvis.show.fitCallbacks(
    trainingSurface,
    ['loss', 'val_loss', 'acc', 'val_acc'],
    { callbacks: ['onBatchEnd', 'onEpochEnd'] }
  )

  const args = {
    epochs: 10,
    // batchSize: 10,
    validationSplit: 0.2,
    callbacks: customCallback
  }

  await model.fit(xs, ys, args)

  trained = true
  trainModelBtn.disabled = false
  xs.dispose()
  ys.dispose()
}

const onSaveModel = () => {
  // TODO
}

const onLoadModel = () => {
  // TODO
}

const onClearTrainingData = () => {
  U.deleteChildren(trainingDataElement)  
}

// TODO: add try/catch/finally
// TODO: disposables
const onMakePredictions = async () => {
  makePredictionsBtn.disabled = true
  U.deleteChildren(predictionsElement)
  const { xs, ys } = generateShapes(10)
  const outputs = model.predict(xs)
  const predictions = await outputs.array()
  const imageTensors = tf.unstack(xs)
  const boundingBoxes = await ys.array()
  const promises = imageTensors.map(async (imageTensor, index) => {
    const actualBoundingBox = boundingBoxes[index]
    const predictedBoundingBox = predictions[index]
    const canvas = await DC.drawImageTensor(predictionsElement, imageTensor)
    DC.drawBoundingBox(canvas, actualBoundingBox, 'blue')
    DC.drawBoundingBox(canvas, predictedBoundingBox, 'red')
  })
  await Promise.all(promises)
  outputs.dispose()
  xs.dispose()
  ys.dispose()
  makePredictionsBtn.disabled = false
}

const onClearPredictions = () => {
  U.deleteChildren(predictionsElement)
  visor && visor.close()
}

const updateButtonStates = () => {
  showVisorBtn.disabled = !visor
  saveModelBtn.disabled = !trained
  makePredictionsBtn.disabled = !trained
  clearTrainingDataBtn.disabled = !trainingDataElement.hasChildNodes()
  clearPredictionsBtn.disabled = !predictionsElement.hasChildNodes()
}

const onIdle = () => {
  updateButtonStates()
  requestAnimationFrame(onIdle)
}

showVisorBtn.addEventListener('click', onShowVisor)
trainModelBtn.addEventListener('click', onTrainModel)
saveModelBtn.addEventListener('click', onSaveModel)
loadModelBtn.addEventListener('click', onLoadModel)
clearTrainingDataBtn.addEventListener('click', onClearTrainingData)
makePredictionsBtn.addEventListener('click', onMakePredictions)
clearPredictionsBtn.addEventListener('click', onClearPredictions)

const main = async () => {
  log.setLevel('info')
  onIdle()
}

main()
