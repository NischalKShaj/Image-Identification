// file to set the routes for the application

// importing the required modules
const express = require("express");
const uploadController = require("../controller/uploadController");
const multer = require("multer");

// setting up the multer
const storage = multer.memoryStorage();
const upload = multer({ storage });

// setting the route
const router = express.Router();

// router for uploading the image for processing
router.post("/uploads", upload.single("image"), uploadController.predict);

// exporting the routes
module.exports = router;
