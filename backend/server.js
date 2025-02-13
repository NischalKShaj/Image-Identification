// creating the server for the application

// importing the required modules
const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");
const corsOptions = require("./config/cors/corsOptions");
dotenv.config();

// setting up the app
const app = express();

// setting up the cors
app.use(cors(corsOptions));

// setting the port
const port = process.env.PORT;

// function for starting the server
const startServer = async () => {
  while (true) {
    try {
      // setting up the routes
      app.use("/", require("./routes/routes"));

      // listing to the port of the application
      app.listen(port, () => {
        console.log(`server running on http://localhost${port}`);
      });
      break;
    } catch (error) {
      console.error("error while connecting", error);
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  }
};

// starting the server
startServer();
