const { spawn } = require('child_process');
const path = require('path');

/**
 * Spawns predict_fake.py to evaluate review text via PyTorch & Ensemble pipeline
 * @param {string} text - Review text
 * @returns {Promise<number>} - Blended probability score (0.0 to 1.0)
 */
const predictFakeScore = (text) => {
  return new Promise((resolve) => {
    // Points to predict_fake.py in root directory
    const scriptPath = path.join(__dirname, '../../predict_fake.py');
    const pythonProcess = spawn('python', [scriptPath, text]);

    let output = '';
    let errorOutput = '';

    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error('Python Execution Error:', errorOutput);
        return resolve(0.15);
      }
      const score = parseFloat(output.trim());
      resolve(isNaN(score) ? 0.15 : score);
    });
  });
};

module.exports = { predictFakeScore };