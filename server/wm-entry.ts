// Entry for the plain-JS bundle used by the server batch (see `npm run build:wm`): the app's own colour-text detector and mask growth,
// so the command line finds overlay lettering exactly like the browser app does.
export { detectChromaText } from '../src/wm/chroma'
export { dilateBinary } from '../src/lib/morphology'
