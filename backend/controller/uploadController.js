// file for setting the controller for the application

// importing the required modules
const dotenv = require("dotenv");
const child = require("child_process");
dotenv.config();

// creating the controller object
const uploadController = {
  // controller for predicting the image
  predict: async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No Image Found" });
      }

      // for setting up the child process for the image prediction
      // const spawn = child.spawn("python", []);
      res.status(200).json({ data: "Success" });
    } catch (error) {
      console.error("error from the predict controller", error);
      res.status(500).json({ error: error });
    }
  },
};

// exporting the modules
module.exports = uploadController;
