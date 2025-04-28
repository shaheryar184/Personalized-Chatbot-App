import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  Picker,
  ScrollView,
} from 'react-native';
import { useRouter } from 'expo-router';

export default function UserInfoScreen() {
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [hobbies, setHobbies] = useState([]);
  const [areaOfInterest, setAreaOfInterest] = useState('');
  const [workingStatus, setWorkingStatus] = useState('');
  const [jobRole, setJobRole] = useState('');
  const [companyName, setCompanyName] = useState('');
  
  const router = useRouter();

  const handleSubmit = () => {
    const userInfo = { name, age, gender, hobbies, areaOfInterest, workingStatus, jobRole, companyName };
    console.log('User Info:', userInfo);
    router.push('/chatU');
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.formContainer}>
        <Text style={styles.title}>User Information</Text>
        
        <TextInput style={styles.input} placeholder="Name" value={name} onChangeText={setName} placeholderTextColor="#bbb" />
        <TextInput style={styles.input} placeholder="Age" keyboardType="numeric" value={age} onChangeText={setAge} placeholderTextColor="#bbb" />
        
        <Text style={styles.label}>Gender</Text>
        <Picker selectedValue={gender} onValueChange={(val) => setGender(val)} style={styles.picker}>
          <Picker.Item label="Select Gender" value="" />
          <Picker.Item label="Male" value="Male" />
          <Picker.Item label="Female" value="Female" />
          <Picker.Item label="Other" value="Other" />
        </Picker>

        <TextInput style={styles.input} placeholder="Hobbies (comma-separated)" value={hobbies.join(', ')} onChangeText={(text) => setHobbies(text.split(', '))} placeholderTextColor="#bbb" />
        
        <Text style={styles.label}>Area of Interest</Text>
        <Picker selectedValue={areaOfInterest} onValueChange={(val) => setAreaOfInterest(val)} style={styles.picker}>
          <Picker.Item label="Select Interest" value="" />
          <Picker.Item label="Tech" value="Tech" />
          <Picker.Item label="Business" value="Business" />
          <Picker.Item label="Health" value="Health" />
          <Picker.Item label="Education" value="Education" />
        </Picker>
        
        <Text style={styles.label}>Working Status</Text>
        <Picker selectedValue={workingStatus} onValueChange={(val) => setWorkingStatus(val)} style={styles.picker}>
          <Picker.Item label="Select Status" value="" />
          <Picker.Item label="Student" value="Student" />
          <Picker.Item label="Professional" value="Professional" />
          <Picker.Item label="Freelancer" value="Freelancer" />
        </Picker>
        
        {workingStatus === 'Professional' && (
          <>
            <TextInput style={styles.input} placeholder="Job Role" value={jobRole} onChangeText={setJobRole} placeholderTextColor="#bbb" />
            <TextInput style={styles.input} placeholder="Company Name" value={companyName} onChangeText={setCompanyName} placeholderTextColor="#bbb" />
          </>
        )}

        <TouchableOpacity style={styles.button} onPress={handleSubmit}>
          <Text style={styles.buttonText}>Submit</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#121212' },
  formContainer: { flex: 1, padding: 20, justifyContent: 'center' },
  title: { fontSize: 26, color: '#fff', marginBottom: 30, textAlign: 'center', fontWeight: 'bold' },
  input: { backgroundColor: '#1e1e1e', padding: 15, borderRadius: 8, color: '#fff', marginBottom: 15, borderWidth: 1, borderColor: '#333' },
  label: { color: '#fff', marginBottom: 5, fontSize: 16, fontWeight: '600' },
  picker: { backgroundColor: '#1e1e1e', color: '#fff', marginBottom: 15, borderRadius: 8, borderWidth: 1, borderColor: '#333' },
  button: { backgroundColor: '#10a37f', padding: 15, borderRadius: 8, alignItems: 'center', marginTop: 10, shadowColor: '#10a37f', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.5, shadowRadius: 4 },
  buttonText: { color: '#fff', fontSize: 18, fontWeight: '700' },
});
