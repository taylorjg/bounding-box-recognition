export const deleteChildren = element => {
  while (element.firstChild) {
    element.removeChild(element.firstChild)
  }
}
