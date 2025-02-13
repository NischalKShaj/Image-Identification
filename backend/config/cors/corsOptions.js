// file to setup the cors for the application

// creating the cors options
const dotenv = require("dotenv");
dotenv.config();

const corsOptions = {
  origin: process.env.BASE_URL,
  methods: ["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS", "HEAD"],
  allowedHeaders: [
    "Origin",
    "X-Requested-With",
    "Content-Type",
    "Accept",
    "Authorization",
  ],
  credentials: true,
};

module.exports = corsOptions;
