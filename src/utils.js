import * as R from 'ramda'

export const deleteChildren = element => {
  while (element.firstChild) {
    element.removeChild(element.firstChild)
  }
}

export const pascalCase = s => `${R.head(s).toUpperCase()}${R.tail(s)}`

export const formatSectionName =
  R.pipe(
    R.split('-'),
    R.map(pascalCase),
    R.join('/')
  )

export const stringFromChars = R.join('')
