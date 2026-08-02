const mongoose = require('mongoose');

const reviewSchema = new mongoose.Schema(
  {
    user: {
      type: String,
      default: 'Anonymous'
    },
    text: {
      type: String,
      required: [true, 'Review text is required'],
      trim: true
    },
    rating: {
      type: Number,
      required: [true, 'Rating is required'],
      min: 1,
      max: 5
    },
    is_flagged: {
      type: Boolean,
      default: false
    },
    fake_score: {
      type: Number,
      default: 0.0
    }
  },
  { timestamps: true }
);

module.exports = mongoose.model('Review', reviewSchema);