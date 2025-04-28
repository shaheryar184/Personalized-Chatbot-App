// app/index.tsx
import { useEffect, useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image } from 'react-native';
import { useRouter } from 'expo-router';

export default function Welcome() {
  const router = useRouter();
  const [showSplash, setShowSplash] = useState(true);

  useEffect(() => {
    // Hide splash screen after 2 seconds
    const timer = setTimeout(() => {
      setShowSplash(false);
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  if (showSplash) {
    return (
      <View style={styles.container}>
        <Image
          source={require('../assets/images/brain-robot.png')}
          style={styles.logo}
        />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.monogram}>
      <Text style={styles.monogramText}>
        AI
      </Text>
      </View>
      
      <Text style={styles.description}>
        Experience the power of AI chat with our AI-style application
      </Text>

      <View style={styles.buttonContainer}>
        <TouchableOpacity 
          style={styles.button}
          onPress={() => router.push('/login')}
        >
          <Text style={styles.buttonText}>Login</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.button, styles.signupButton]}
          onPress={() => router.push('/signup')}
        >
          <Text style={styles.buttonText}>Sign Up</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  logo: {
    width: 150,
    height: 150,
    resizeMode: 'contain',
  },
  monogram: {
    width: 100,
    height: 100,
    backgroundColor: '#10a37f',
    borderRadius: 50,
    marginBottom: 30,
  },
  monogramText: {
    color: '#fefe',
    fontSize: 50,
    fontWeight: 'bold',
    textAlign: 'center',
    marginTop: 20,
  },
  description: {
    color: '#fff',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 40,
  },
  buttonContainer: {
    width: '100%',
    gap: 15,
  },
  button: {
    backgroundColor: '#10a37f',
    padding: 15,
    borderRadius: 5,
    alignItems: 'center',
  },
  signupButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#10a37f',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});