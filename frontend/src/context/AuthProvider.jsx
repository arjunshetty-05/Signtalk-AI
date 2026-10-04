// AuthProvider.jsx — SignTalk AI web dashboard
//
// Firebase Auth email/password login. Stores the current user + ID token in
// context and attaches the token as a Bearer header on every Axios call via
// a request interceptor, so components never need to manually pass it.

import { createContext, useContext, useEffect, useState } from "react";
import {
  onIdTokenChanged,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut as firebaseSignOut,
} from "firebase/auth";
import axios from "axios";
import { auth, API_BASE_URL } from "../firebase.js";

const AuthContext = createContext(null);

export const apiClient = axios.create({ baseURL: API_BASE_URL });

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // onIdTokenChanged (not onAuthStateChanged) fires on sign-in/out AND on
    // Firebase's automatic hourly token refresh, so `token` state — and
    // anything reconnecting off of it, like useGestureSocket — always has
    // a live token instead of silently going stale after ~1 hour.
    const unsubscribe = onIdTokenChanged(auth, async (firebaseUser) => {
      if (firebaseUser) {
        const idToken = await firebaseUser.getIdToken();
        setUser(firebaseUser);
        setToken(idToken);
      } else {
        setUser(null);
        setToken(null);
      }
      setLoading(false);
    });
    return unsubscribe;
  }, []);

  useEffect(() => {
    const interceptorId = apiClient.interceptors.request.use(async (config) => {
      if (auth.currentUser) {
        const freshToken = await auth.currentUser.getIdToken();
        config.headers.Authorization = `Bearer ${freshToken}`;
      }
      return config;
    });
    return () => apiClient.interceptors.request.eject(interceptorId);
  }, []);

  const login = (email, password) => signInWithEmailAndPassword(auth, email, password);
  const signup = (email, password) => createUserWithEmailAndPassword(auth, email, password);
  const logout = () => firebaseSignOut(auth);

  const value = { user, token, loading, login, signup, logout };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
