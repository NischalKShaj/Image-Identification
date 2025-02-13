// file to set the routes for the application

// importing the required modules
const express = require("express");

// setting the route
const router = express.Router();

// router for uploading the image for processing
router.post("/uploads");

// exporting the routes
module.exports = router;
