import { createContext, useContext, useEffect, useState, useCallback } from "react";
import { Alert } from "react-native";
import * as WebBrowser from "expo-web-browser";
import * as Linking from "expo-linking";
import { Session, User } from "@supabase/supabase-js";
import { supabase } from "../lib/supabase";

WebBrowser.maybeCompleteAuthSession();

const REDIRECT_URL = Linking.createURL("auth-callback");

type AuthContextType = {
  session: Session | null;
  user: User | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (name: string, email: string, password: string) => Promise<void>;
  signInWithGoogle: () => Promise<void>;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    supabase.auth.getSession()
      .then(({ data }) => setSession(data.session))
      .catch(() => setSession(null))
      .finally(() => setLoading(false));

    const { data: listener } = supabase.auth.onAuthStateChange((_event, newSession) => {
      setSession(newSession);
      setLoading(false);
    });

    return () => listener.subscription.unsubscribe();
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) throw error;
  }, []);

  const signUp = useCallback(async (name: string, email: string, password: string) => {
    const { error } = await supabase.auth.signUp({
      email,
      password,
      options: { data: { full_name: name } },
    });
    if (error) throw error;
  }, []);

  const signInWithGoogle = useCallback(async () => {
    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: REDIRECT_URL, skipBrowserRedirect: true },
    });
    if (error) throw error;
    if (!data.url) throw new Error("No OAuth URL returned");

    const result = await WebBrowser.openAuthSessionAsync(data.url, REDIRECT_URL);
    if (result.type !== "success") return;

    const parsed = new URL(result.url);
    const code = parsed.searchParams.get("code");
    if (code) {
      const { error: exchangeError } = await supabase.auth.exchangeCodeForSession(result.url);
      if (exchangeError) throw exchangeError;
      return;
    }

    const hash = new URLSearchParams(parsed.hash.slice(1));
    const accessToken = hash.get("access_token");
    const refreshToken = hash.get("refresh_token");
    if (accessToken && refreshToken) {
      const { error: sessionError } = await supabase.auth.setSession({
        access_token: accessToken,
        refresh_token: refreshToken,
      });
      if (sessionError) throw sessionError;
    }
  }, []);

  const signOut = useCallback(async () => {
    const { error } = await supabase.auth.signOut();
    if (error) Alert.alert("Error", error.message);
  }, []);

  return (
    <AuthContext.Provider
      value={{
        session,
        user: session?.user ?? null,
        loading,
        signIn,
        signUp,
        signInWithGoogle,
        signOut,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside AuthProvider");
  return context;
}
