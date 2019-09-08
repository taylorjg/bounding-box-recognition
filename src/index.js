import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import log from 'loglevel'
import * as C from './constants'
import * as U from './utils'
import * as DC from './drawCanvas'
import { generateShapes } from './shapes'
import { showErrorPanel, hideErrorPanel } from './errorPanel'

let model = undefined
let training = false
let predicting = false
let trained = false
let visor = undefined

const reportMemory = () => {
  log.info(`tf memory: ${JSON.stringify(tf.memory())}`)
}

const yieldToEventLoop = () =>
  new Promise(resolve => setTimeout(resolve, 0))

const getVisor = () => {
  if (!visor) {
    visor = tfvis.visor()
  }
  visor.open()
  return visor
}

const showVisorBtn = document.getElementById('show-visor-btn')
const trainModelBtn = document.getElementById('train-model-btn')
// const saveModelBtn = document.getElementById('save-model-btn')
// const loadModelBtn = document.getElementById('load-model-btn')
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

const onTrainModel = async () => {
  const disposables = []
  try {
    hideErrorPanel()
    trained = false
    training = true
    await yieldToEventLoop()

    U.deleteChildren(trainingDataElement)
    await yieldToEventLoop()

    model && model.dispose()

    const cb = async (shape, index) => {
      if (index >= 10) return
      const canvas = await DC.drawImageTensor(trainingDataElement, shape.imageTensor)
      DC.drawBoundingBox(canvas, shape.boundingBox, 'blue')
    }
    const { imageTensors, xs, ys } = await generateShapes(100, cb)
    disposables.push(...imageTensors, xs, ys)
    await yieldToEventLoop()

    model = createModel() // eslint-disable-line

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
      batchSize: 25,
      validationSplit: 0.2,
      callbacks: customCallback
    }

    await model.fit(xs, ys, args)
    trained = true
  } catch (error) {
    log.error(`[onTrainModel] ${error.message}`)
    showErrorPanel(error.message)
  } finally {
    disposables.forEach(disposable => disposable.dispose())
    reportMemory()
    training = false
  }
}

// const onSaveModel = () => {
//   try {
//     hideErrorPanel()
//     // TODO
//   } catch (error) {
//     log.error(`[onSaveModel] ${error.message}`)
//     showErrorPanel(error.message)
//   } finally {
//     reportMemory()
//   }
// }

// const onLoadModel = () => {
//   try {
//     hideErrorPanel()
//     model && model.dispose()
//     // TODO
//   } catch (error) {
//     log.error(`[onLoadModel] ${error.message}`)
//     showErrorPanel(error.message)
//   } finally {
//     reportMemory()
//   }
// }

const onClearTrainingData = () => {
  U.deleteChildren(trainingDataElement)
  model && model.dispose()
  trained = false
  reportMemory()
}

const onMakePredictions = async () => {
  const disposables = []
  try {
    hideErrorPanel()
    predicting = true
    await yieldToEventLoop()

    U.deleteChildren(predictionsElement)
    await yieldToEventLoop()

    const { imageTensors, xs, ys } = await generateShapes(10)
    disposables.push(...imageTensors, xs, ys)

    const outputs = model.predict(xs)
    disposables.push(outputs)
    const predictions = await outputs.array()
    const boundingBoxes = await ys.array()

    const promises = imageTensors.map(async (imageTensor, index) => {
      const actualBoundingBox = boundingBoxes[index]
      const predictedBoundingBox = predictions[index]
      const canvas = await DC.drawImageTensor(predictionsElement, imageTensor)
      DC.drawBoundingBox(canvas, actualBoundingBox, 'blue')
      DC.drawBoundingBox(canvas, predictedBoundingBox, 'red')
    })
    await Promise.all(promises)
  } catch (error) {
    log.error(`[onMakePredictions] ${error.message}`)
    showErrorPanel(error.message)
  } finally {
    disposables.forEach(disposable => disposable.dispose())
    reportMemory()
    predicting = false
  }
}

const onClearPredictions = () => {
  U.deleteChildren(predictionsElement)
  visor && visor.close()
}

showVisorBtn.addEventListener('click', onShowVisor)
trainModelBtn.addEventListener('click', onTrainModel)
// saveModelBtn.addEventListener('click', onSaveModel)
// loadModelBtn.addEventListener('click', onLoadModel)
clearTrainingDataBtn.addEventListener('click', onClearTrainingData)
makePredictionsBtn.addEventListener('click', onMakePredictions)
clearPredictionsBtn.addEventListener('click', onClearPredictions)

const updateButtonStates = () => {
  showVisorBtn.disabled = !visor
  // saveModelBtn.disabled = !trained
  trainModelBtn.disabled = training
  makePredictionsBtn.disabled = !trained || predicting
  clearTrainingDataBtn.disabled = !trainingDataElement.hasChildNodes() || training
  clearPredictionsBtn.disabled = !predictionsElement.hasChildNodes() || predicting
}

const onIdle = () => {
  updateButtonStates()
  requestAnimationFrame(onIdle)
}

const main = async () => {
  log.setLevel('info')
  onIdle()
}

main()
