const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '.env') });
require('dotenv').config({ path: path.join(__dirname, '../.env') });

const express = require('express');
const mongoose = require('mongoose');

const Review = require('./models/Review');
const { predictFakeScore } = require('./services/fakeDetectorService');

const app = express();
app.use(express.json());

// MongoDB Atlas Connection
mongoose
  .connect(process.env.MONGO_URI)
  .then(() => console.log('✅ Connected to MongoDB Atlas (revx_db) successfully!'))
  .catch((err) => console.error('❌ MongoDB Connection Error:', err));

// Threshold aligned with ensemble_pipeline target (0.60)
const FAKE_THRESHOLD = parseFloat(process.env.FAKE_THRESHOLD) || 0.60;

/**
 * @route   POST /api/reviews
 * @desc    Submit review -> Execute ML Pipeline -> Block (422) or Save (201)
 */
app.post('/api/reviews', async (req, res) => {
  try {
    const { user, text, rating } = req.body;

    if (!text || !rating) {
      return res.status(400).json({
        success: false,
        message: 'Please provide both review text and rating.'
      });
    }

    // 1. Calculate score via PyTorch + Ensemble pipeline
    const fakeScore = await predictFakeScore(text);
    const isFlagged = fakeScore > FAKE_THRESHOLD;

    // 2. Reject if score exceeds decision threshold
    if (isFlagged) {
      return res.status(422).json({
        success: false,
        error_code: 'FLAGGED_FAKE_REVIEW',
        message: 'Submission rejected: High probability of automated or fake review text.',
        details: {
          fake_score: fakeScore,
          threshold: FAKE_THRESHOLD
        }
      });
    }

    // 3. Save approved review to MongoDB Atlas
    const newReview = await Review.create({
      user: user || 'Anonymous',
      text,
      rating,
      is_flagged: false,
      fake_score: fakeScore
    });

    return res.status(201).json({
      success: true,
      message: 'Review saved successfully.',
      data: newReview
    });

  } catch (error) {
    console.error('Error processing review:', error);
    return res.status(500).json({
      success: false,
      message: 'Server error processing review.',
      error: error.message
    });
  }
});

/**
 * @route   GET /api/reviews
 * @desc    Fetch reviews for Admin view
 */
app.get('/api/reviews', async (req, res) => {
  try {
    const reviews = await Review.find().sort({ createdAt: -1 });
    return res.status(200).json({
      success: true,
      count: reviews.length,
      data: reviews
    });
  } catch (error) {
    return res.status(500).json({ success: false, error: error.message });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});