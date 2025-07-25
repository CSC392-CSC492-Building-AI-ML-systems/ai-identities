import { MongoClient, Db } from "mongodb";

// load env variables from .env file
const MONGODB_URI = process.env.MONGODB_URI || "mongodb://localhost:27017";
const MONGODB_DB = process.env.MONGODB_DB || "ai-identities";

// cache the mongodb client so we dont have to connect to db everytime
let client: MongoClient | undefined;

// function that gets the mongodb
export async function getDatabase(): Promise<Db> {
  if (process.env.NODE_ENV === "development") {
    // if client is not initialized, initialize it3
    if (!client) {
      client = await new MongoClient(MONGODB_URI).connect();
    }
    return client.db(MONGODB_DB);
  } else {
    throw new Error("Production mode not implemented yet.");
  }
}
