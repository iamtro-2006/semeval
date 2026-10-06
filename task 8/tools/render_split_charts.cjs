// Convert the saved vector charts to PNG; all caches stay in the workspace.
const path = require('path');
const fs = require('fs');
const output = path.resolve(__dirname, '../outputs/en_split');
if (process.platform === 'win32') {
  process.env.FONTCONFIG_FILE = path.join(output, 'fontconfig.xml');
  fs.mkdirSync(path.join(output, '.fontcache'), {recursive: true});
}
const sharp = require('C:/Users/ASUS/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
Promise.all(['distribution_overview', 'target_labels', 'annotation_labels'].map(name =>
  sharp(path.join(output, name + '.svg')).resize({width: 1600}).png()
    .toFile(path.join(output, name + '.png'))
)).then(() => process.stdout.write('Rendered 3 charts.\n')).catch(error => {
  process.stderr.write(String(error));
  process.exitCode = 1;
});
