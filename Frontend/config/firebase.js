
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {

    // secret key for firebase
  };
const app = initializeApp(firebaseConfig);
export const authDB = getAuth(app);



