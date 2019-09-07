const path = require('path')
const express = require('express')
const morgan = require('morgan')
const log = require('loglevel')

const PORT = process.env.PORT || 3070
const DIST_FOLDER = path.resolve(__dirname, '..', 'dist')

log.setLevel('info')

const app = express()
app.use(morgan('dev'))
app.use(express.static(DIST_FOLDER))
app.listen(PORT, () => log.info(`Listening on http://localhost:${PORT}`))
